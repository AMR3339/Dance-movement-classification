# 舞蹈动作细粒度分类复现项目（MS-G3D + 曲率增强 + 节拍对齐 + 原型对比）

复现论文《Fine-Grained Classification of Dance Movements Using MS-G3D Network》中的**本文方法**：
以 MS-G3D 为骨干，在骨架序列输入侧引入**离散曲率算子增强**与**节拍对齐**预处理，在特征嵌入侧引入
**动量原型对比损失**与**温度缩放度量头**，实现舞蹈动作 24 子类细粒度分类。

由于本机无真实 AIST++ 数据（需 Google Research 下载，网络不可达），本项目默认使用
**合成舞蹈骨架数据**（模拟 24 子类 × 24 关节 × 3 通道 × 64 帧，含舞者节拍偏移与噪声）
完整跑通方法全流程，各阶段中间结果（表格/图）保存于 `outputs/`。

## 方法（本文方法）

| 模块 | 说明 | 公式 |
|---|---|---|
| 曲率增强 | 离散曲率算子放大末端关节（腕/踝）高频微变形，沿切向位移补偿 | 式(5)(6) |
| 节拍对齐 | 运动能量包络 + 自相关周期 + 互相关相位偏差 + 三次样条时间重映射 | 式(7)(8) |
| 自适应图 | 行归一化 + 余弦相似度 + 图温度 Softmax，与物理邻接凸组合 | 式(2)(3)(4) |
| MS-G3D | 多尺度图聚合（1-hop/2-hop/跳跃子图）+ 膨胀时序卷积（9 层，dilation [1,1,2,1,2,4,2,4,1]） | 式(1) |
| 原型对比 | 动量原型更新 m=0.99 + 温度缩放 τ=0.2 + 困难负样本挖掘 k=3 | 式(9)(10) |
| 联合损失 | 加权 CE + λ_proto·L_proto + λ_reg·‖A_adapt‖_F | 式(11) |

## 快速开始

```bash
py -m pip install -r requirements.txt
py run_all.py                  # 一键跑通全流程（含 10 epochs 训练，默认 CPU）
py run_all.py --skip-train     # 跳过训练，使用已有权重做评估与可视化
```

## 目录结构

```
dance_recognizer/
├── run_all.py            # 一键全流程入口
├── config.yaml           # 全局配置（数据/预处理/模型/损失/训练/评估）
├── data/                 # 24 关节拓扑 / 合成数据生成 / AIST++ 加载 / 预处理管线
├── models/               # MS-G3D 骨干 + 自适应图 + 原型对比损失
├── engine/               # 训练器（AdamW+两阶段调度+GSNR）/ 评估器
├── viz/                  # 各阶段可视化
├── smoke_test.py         # 冒烟测试
└── outputs/              # 所有结果（按 stage 组织）
    ├── stage0_data_stats/    # 类别分布 + 关节轨迹样例
    ├── stage1_preprocess/    # 预处理对比：轨迹 / 曲率分布 / 能量包络 / 节拍模板
    ├── stage2_graph/         # 自适应邻接矩阵热图（式2-4）
    ├── stage3_training/      # 损失+GSNR 双轴 / 原型 PCA 轨迹 / LR 调度
    ├── stage4_eval/          # 混淆矩阵 / 类别难度 / t-SNE 嵌入 / 指标 JSON
    ├── weights/              # best.pt / history.json / 原型
    └── REPORT.md             # 最终结果汇总报告（自动生成）
```

## 各阶段中间结果（对应论文 §3.1-3.4 / §4）

| 阶段 | 内容 | 输出 |
|---|---|---|
| Stage 0 | 数据构建：24 子类合成骨架 + dancer-independent 划分 | `stage0_data_stats/` |
| Stage 1 | 曲率增强（式5/6）+ 节拍对齐（式7/8）预处理与对比图 | `stage1_preprocess/` |
| Stage 2 | MS-G3D 骨干 + 自适应图（式1-4）邻接矩阵可视化 | `stage2_graph/` |
| Stage 3 | 训练：加权 CE + 原型对比（式9-11）+ GSNR 记录 | `stage3_training/` |
| Stage 4 | 评估：Top-1 / Macro F1 / Cross-Pair F1 / FDI（式12）/ 聚类指标 | `stage4_eval/` |

## 复现结果（10 epochs，合成数据测试集）

| 指标 | 复现值 | 论文报道值 |
|---|---|---|
| Top-1 Acc | **93.9%** | 89.7% |
| Macro F1 | **94.8%** | 88.3% |
| Cross-Pair F1 | **100.0%** | 84.6% |
| FDI | 0.737 | 0.41 |
| 平均轮廓系数 | 0.494 | 0.342 |

> 合成数据类间基元差异明确，故数值与论文同量级且略高；真实 AIST++ 数据的性能以论文为准。

## 真实数据接入（可选）

1. 下载 AIST++（Google Research）放入 `data_cache/AIST_DANCE/data`；
2. 准备 24 细分类标注 CSV（`sequence_id,label`）；
3. 修改 `config.yaml`：`data.source: "aistpp"`、`data.aistpp.path`、`model.n_frames: 128`、`model.channels: [64,128,256]`；
4. 重新运行 `py run_all.py`（同一套代码，无需改动模型）。
