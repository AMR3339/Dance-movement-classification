"""MS-G3D 骨干 + 自适应图学习分支 + 温度缩放度量头（论文 §3.1）。

- 多尺度图聚合（式1）：Y = sum_k lambda_k · (A_k_norm @ X) @ W_k + b_k
    尺度 k：1-hop 物理边 / 2-hop 跨肢 / 跳跃子图；lambda_k 可学习融合权重（初始 1/3）。
- 自适应图学习分支（式2-4）：
    式2 行归一化：F_norm = F / ||F||（单位向量）
    式3 余弦相似度：S = F_norm @ F_norm^T
    式4 图温度 Softmax：A_adapt = Softmax(S / tau)
    与物理邻接矩阵做凸组合后送入每块图聚合步骤（块内算子不修改）。
- 时序膨胀卷积堆叠：dilation 序列 [1,1,2,1,2,4,2,4,1]，感受野 7 -> 145 帧。
- 度量头：全局平均池化 -> D=256 嵌入；温度缩放余弦分类 logits = cos(z, P_c)/tau。
布局：模型内部 (B, C, T, V)，输入 (B, V, C, T)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.topology import build_adjacency, normalized_adjacency
import numpy as np


class SelfAttentionGraph(nn.Module):
    """式(2)(3)(4)：数据驱动自适应邻接矩阵。"""

    def __init__(self, channels, graph_tau=0.2):
        super().__init__()
        self.tau = graph_tau
        self.proj = nn.Conv2d(channels, channels, 1)   # 节点特征投影（轻量）
        self.alpha = nn.Parameter(torch.tensor(0.5))   # 自适应/物理凸组合权重

    def forward(self, x):
        """x: (B, C, T, V) -> A_adapt: (B, V, V)"""
        B, C, T, V = x.shape
        f = self.proj(x).mean(dim=2)                   # (B, C, V) 节点级特征
        f = f.permute(0, 2, 1)                         # (B, V, C)
        # 式(2) 行归一化
        f_norm = F.normalize(f, dim=-1)
        # 式(3) 余弦相似度
        S = torch.bmm(f_norm, f_norm.transpose(1, 2))
        # 式(4) 图温度 Softmax
        A = F.softmax(S / self.tau, dim=-1)
        return A


class MSG3DBlock(nn.Module):
    """MS-G3D 时空块：多尺度图卷积（式1）+ 膨胀时序卷积 + 残差 + 自适应图。"""

    def __init__(self, in_channels, out_channels, A_phys, dilation,
                 tcn_kernel=3, graph_tau=0.2, stride=1, use_adaptive=True):
        super().__init__()
        self.num_scales = len(A_phys)
        self.use_adaptive = use_adaptive
        # 物理多尺度邻接（归一化，注册为 buffer）
        A_norm = torch.from_numpy(np.stack([normalized_adjacency(a) for a in A_phys])).float()
        self.register_buffer("A_phys", A_norm)          # (S, V, V)
        # 尺度融合权重（式1 lambda_k，初始均匀 1/3，可学习）
        self.lambda_k = nn.Parameter(torch.ones(self.num_scales) / self.num_scales)
        # 尺度卷积核（式1 W_k）
        self.gcn_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, 1) for _ in range(self.num_scales)])
        # 时序膨胀卷积
        pad = (tcn_kernel - 1) * dilation // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (1, tcn_kernel),
                      padding=(0, pad), dilation=(1, dilation)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        if use_adaptive:
            self.adapt = SelfAttentionGraph(in_channels, graph_tau)
        # 残差
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        """x: (B, C, T, V) -> (B, C', T, V)；返回 (out, A_adapt)"""
        B, C, T, V = x.shape
        A_adapt = self.adapt(x) if self.use_adaptive else None
        out = 0.0
        lam = F.softmax(self.lambda_k, dim=0)
        for k in range(self.num_scales):
            A_k = self.A_phys[k]
            if A_adapt is not None:
                alpha = torch.sigmoid(self.adapt.alpha)
                A_used = alpha * A_adapt + (1 - alpha) * A_k.unsqueeze(0)
            else:
                A_used = A_k.unsqueeze(0)
            # 式(1)：A_k @ X（关节维）
            xa = torch.einsum("btcv,bvw->btcw", x, A_used)
            out = out + lam[k] * self.gcn_convs[k](xa)
        out = self.tcn(out) + self.residual(x)
        return out, A_adapt


class MSG3D(nn.Module):
    """MS-G3D 骨干 + 度量头。

    输入 (B, V, C_in, T)；输出 (logits_metric, embedding, adapt_mats)。
    """

    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.n_joints = m["n_joints"]
        self.emb_dim = m["emb_dim"]
        self.num_classes = m["num_classes"]
        self.graph_tau = m["graph_tau"]
        self.A_phys = build_adjacency(self.n_joints)
        channels = m["channels"]
        dilations = m["dilations"]
        block_nums = m["block_nums"]
        assert sum(block_nums) == len(dilations) == 9, "9 层级联块"

        # 输入映射：1×1 线性卷积
        self.input_map = nn.Sequential(
            nn.Conv2d(m["in_channels"], channels[0], 1),
            nn.BatchNorm2d(channels[0]), nn.ReLU(inplace=True))

        self.blocks = nn.ModuleList()
        ch_in = channels[0]
        d_idx = 0
        for stage, n_blk in enumerate(block_nums):
            ch_out = channels[stage]
            for i in range(n_blk):
                self.blocks.append(MSG3DBlock(
                    ch_in, ch_out, self.A_phys, dilation=dilations[d_idx],
                    tcn_kernel=m["tcn_kernel"], graph_tau=self.graph_tau,
                    use_adaptive=m["adaptive_graph"]))
                ch_in = ch_out
                d_idx += 1

        # 嵌入头：全局平均池化 -> D=256
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.embed = nn.Sequential(
            nn.Linear(channels[-1], self.emb_dim),
            nn.BatchNorm1d(self.emb_dim), nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(m["dropout"])

    def forward(self, x, prototypes=None, tau_metric=0.2):
        """x: (B, V, C, T)。返回 (logits_metric, embedding, adapt_mats, logits_linear)。"""
        x = x.permute(0, 2, 3, 1)                       # (B, C, T, V)
        x = self.input_map(x)
        adapt_mats = []
        for blk in self.blocks:
            x, A_adapt = blk(x)
            adapt_mats.append(A_adapt)
        z = self.gap(x).flatten(1)                      # (B, C_last)
        z = self.dropout(z)
        emb = self.embed(z)                             # (B, D=256)
        # 温度缩放度量头：logits = cos(z, P_c) / tau（原型相似度分类）
        if prototypes is not None and prototypes.any():
            prot_norm = F.normalize(prototypes, dim=-1)
            z_norm = F.normalize(emb, dim=-1)
            logits_metric = torch.mm(z_norm, prot_norm.t()) / tau_metric
        else:
            logits_metric = None
        return logits_metric, emb, adapt_mats
