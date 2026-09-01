"""原型对比损失与联合目标（论文 §3.3，式(9)(10)(11)）。

式(9) 动量原型更新：P_c <- m·P_c + (1-m)·mean(z_i)，m=0.99，保持历史平滑；
式(10) 原型对比损失：基于负欧氏距离 + 温度缩放 + 困难负样本挖掘（top-k 最近异类原型）；
式(11) 联合目标：L = CE(weighted) + lambda_proto·L_proto + lambda_reg·||A_adapt||_F。
"""
import torch
import torch.nn.functional as F


class PrototypeMemory:
    """类原型记忆库（式9 动量更新）。"""

    def __init__(self, num_classes: int, emb_dim: int, momentum: float = 0.99, device="cpu"):
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.momentum = momentum
        self.prototypes = torch.zeros(num_classes, emb_dim, device=device)
        self.counts = torch.zeros(num_classes, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, z: torch.Tensor, labels: torch.Tensor):
        """式(9)：P_c <- m·P_c + (1-m)·mean_batch(z_c)"""
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                mean = z[mask].mean(dim=0)
                if self.counts[c] == 0:
                    self.prototypes[c] = mean
                else:
                    self.prototypes[c] = self.momentum * self.prototypes[c] + \
                        (1 - self.momentum) * mean
                self.counts[c] += 1

    def get(self):
        return self.prototypes


def prototype_contrast_loss(z: torch.Tensor, labels: torch.Tensor, prototypes: torch.Tensor,
                            tau: float = 0.2, k: int = 3):
    """式(10)：温度缩放原型对比损失 + 困难负样本挖掘。

    L = -log( exp(-d(z_i, P_y)/tau) / (exp(-d(z_i,P_y)/tau) + sum_{j in H_i} exp(-d(z_i,P_j)/tau)) )
    H_i 为与 z_i 欧氏距离最小的 k 个异类原型（困难负样本）。
    """
    dist = torch.cdist(z, prototypes, p=2)              # (B, K) 欧氏距离
    logits = -dist / tau                                 # 温度缩放
    B = z.shape[0]
    pos = logits[torch.arange(B, device=z.device), labels]
    neg_mask = torch.ones_like(logits, dtype=torch.bool)
    neg_mask[torch.arange(B, device=z.device), labels] = False
    masked = logits.masked_fill(~neg_mask, float("-inf"))
    hard, _ = torch.topk(masked, k, dim=1)               # 困难负样本 top-k
    denom = torch.logsumexp(torch.cat([pos.unsqueeze(1), hard], dim=1), dim=1)
    return -(pos - denom).mean()


def class_weights(labels: torch.Tensor, num_classes: int):
    """加权交叉熵类别权重（逆频率，抑制长尾）。"""
    counts = torch.bincount(labels, minlength=num_classes).float() + 1e-6
    w = counts.sum() / (num_classes * counts)
    return w / w.sum() * num_classes


def frobenius_reg(adapt_mats):
    """式(11)：自适应邻接矩阵的 Frobenius 范数正则（平滑拓扑权重分布）。"""
    norms = [a.norm() for a in adapt_mats if a is not None]
    return sum(norms) / len(norms) if norms else torch.tensor(0.0)


def compute_loss(logits_metric, labels, emb, prototypes, adapt_mats, loss_cfg,
                 ce_weights=None):
    """式(11) 联合损失。返回 (total, components_dict)。"""
    l_ce = F.cross_entropy(logits_metric, labels, weight=ce_weights)
    l_proto = prototype_contrast_loss(emb, labels, prototypes,
                                      tau=loss_cfg["proto_temp"],
                                      k=loss_cfg["hard_negatives"])
    l_reg = frobenius_reg(adapt_mats)
    total = l_ce + loss_cfg["lambda_proto"] * l_proto + loss_cfg["lambda_reg"] * l_reg
    return total, {"ce": l_ce.item(), "proto": l_proto.item(),
                   "reg": l_reg.item() if isinstance(l_reg, torch.Tensor) else l_reg}
