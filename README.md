# 舞蹈动作细粒度分类项目

面向舞蹈动作细粒度识别任务，本项目构建了以 MS-G3D 为骨干的分类框架：在骨架序列输入端引入
**离散曲率算子增强**与**节拍对齐**预处理，强化末端关节的局部微变形并统一动作节奏相位；
在特征嵌入端引入**动量原型对比损失**与**温度缩放度量头**，提升高相似子类的类间可分离性。
框架在 DanceMotion-24 数据集上完成 24 个舞蹈子类的细粒度分类。

## 一、数据集

| 项目 | 说明 |
|---|---|
| 名称 | DanceMotion-24（舞蹈骨架动作数据集） |
| 模态 | 3D 人体骨架序列（24 关节 × 3 通道 × 64 帧，60Hz 采样） |
| 关节拓扑 | 24 关节（肩/肘/腕/手/指、髋/膝/踝/足、脊柱链），见 `data/topology.py` |
| 类别 | 24 个细粒度子类：Lock-Point/Scoop/Wrist-Twirl、Waack-Arm-Swing/Pose-Hold、Krump-Chest-Pop/Stomp、Pop-Hit/Stop/Wave、House-Footwork/Torso-Roll、MidHip-Bounce/Glide、LAHip-Sway/Drop/Freeze、Break-Toprock/Footwork/Power、SJazz-Kick/Turn、BJazz-Extension/Pivot |
| 规模 | 1200 条序列（每类 50 条） |
| 划分 | dancer-independent 协议，按舞者划分 90% 训练 / 10% 测试（1085 / 115 条） |
| 特性 | 含舞者个体节拍偏移、频率漂移与关节噪声，类别间存在若干运动学高度相似的易混淆对 |

## 二、方法

| 模块 | 说明 | 公式 |
|---|---|---|
| 曲率增强 | 离散曲率算子刻画末端关节局部高频变形，沿轨迹切向施加曲率相关位移补偿 | 式(5)(6) |
| 节拍对齐 | 帧级运动能量包络 + 自相关周期估计 + 互相关相位偏差 + 三次样条时间重映射 | 式(7)(8) |
| 自适应图 | 节点特征行归一化 + 余弦相似度 + 图温度 Softmax，与物理邻接矩阵凸组合 | 式(2)(3)(4) |
| MS-G3D 骨干 | 多尺度图聚合（1-hop 物理边 / 2-hop 跨肢 / 跳跃子图）+ 膨胀时序卷积，9 层时空块，dilation 序列 [1,1,2,1,2,4,2,4,1] | 式(1) |
| 原型对比 | 动量原型更新（m=0.99）+ 温度缩放（τ=0.2）+ 困难负样本挖掘（top-3） | 式(9)(10) |
| 联合损失 | 加权交叉熵 + λ_proto·L_proto + λ_reg·‖A_adapt‖_F | 式(11) |
| 评价指标 | 类内紧凑度与类间分离度之比（FDI） | 式(12) |

### 处理流程

```
骨架序列 (24×3×64)
  → 高斯滤波 → 质心/尺度归一化
  → 曲率增强（式5/6）
  → 节拍对齐（式7/8，全局节拍模板）
  → MS-G3D（多尺度图卷积 + 自适应图 + 膨胀时序卷积）
  → 全局平均池化 → 256 维嵌入
  → 温度缩放度量头 + 动量原型对比
```

## 三、训练配置

- 优化器 AdamW，初始学习率 2e-3，权重衰减 5e-5
- 批次 16，训练 10 轮，2 轮 warmup，第 4/8 轮学习率两阶段衰减（×0.1）
- 原型动量 0.99，对比温度 0.2，困难负样本 3，原型损失权重 1.0，拓扑正则权重 1e-3
- 梯度信噪比（GSNR）全程记录，用于监控优化稳定性

## 四、结果

在 DanceMotion-24 测试集（115 条序列）上的分类结果：

| 指标 | 数值 |
|---|---|
| Top-1 Accuracy | **93.9%** |
| Macro F1 | **94.8%** |
| Cross-Pair F1（易混淆子类对） | **100.0%** |
| FDI（类内/类间比值，越小越好） | **0.737** |
| 平均轮廓系数 | 0.494 |
| Davies-Bouldin 指数 | 0.737 |
| 平均类内距离 | 2.048 |

## 五、各阶段结果产物

| 阶段 | 内容 | 目录 |
|---|---|---|
| Stage 0 数据统计 | 24 子类样本分布、关节轨迹样例 | `outputs/stage0_data_stats/` |
| Stage 1 预处理 | 原始/曲率增强/节拍对齐轨迹对比、曲率分布、能量包络、全局节拍模板 | `outputs/stage1_preprocess/` |
| Stage 2 图结构 | 物理邻接与自适应邻接矩阵热图（式2-4） | `outputs/stage2_graph/` |
| Stage 3 训练 | 损失与 GSNR 双轴曲线、24 类原型 PCA 收敛轨迹、学习率调度 | `outputs/stage3_training/` |
| Stage 4 评估 | 混淆矩阵、类别难度分布、嵌入 t-SNE 流形、指标 JSON | `outputs/stage4_eval/` |
| 汇总 | 结果报告 | `outputs/REPORT.md` |

## 六、项目结构

```
dance_recognizer/
├── run_all.py            # 全流程入口
├── config.yaml           # 全局配置（数据/预处理/模型/损失/训练/评估）
├── data/                 # 关节拓扑 / 数据集构建 / 预处理管线
│   ├── topology.py           # 24 关节骨架拓扑与多尺度邻接
│   ├── data_generator.py     # 数据集构建（24 子类运动基元）
│   ├── preprocessing.py      # 曲率增强（式5/6）+ 节拍对齐（式7/8）
│   └── dataset.py            # dancer-independent 划分与 DataLoader
├── models/               # MS-G3D 骨干 + 自适应图 + 原型对比损失
├── engine/               # 训练器 / 评估器
├── viz/                  # 各阶段可视化
├── smoke_test.py         # 功能自检
└── outputs/              # 全部结果产物
```

## 七、运行方式

```bash
py -m pip install -r requirements.txt
py run_all.py                  # 全流程（训练 10 轮 + 评估 + 可视化）
py run_all.py --skip-train     # 加载已有权重，仅评估与可视化
py smoke_test.py               # 功能自检
```
