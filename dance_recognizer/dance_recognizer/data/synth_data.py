"""合成舞蹈骨架数据生成器。

在没有真实 AIST++ 数据的情况下，按论文描述模拟 24 个细粒度舞蹈子类的骨架序列：
  - 24 关节 × 3 通道 × 128 帧（60Hz 采样，时长约 2.1s）
  - 每类定义运动基元（主动关节组 + 运动模式 + 频率 + 幅度），类间差异聚焦末端关节
    微变形（腕/踝/髋），并刻意构造易混淆类对（如 Lock-Wrist-Twirl vs Waack-Arm-Swing）
  - 类内引入舞者个体节拍偏移（beat jitter）、频率漂移与高斯噪声，模拟节奏语义纠缠
  - 按舞者 ID 做 dancer-independent 划分（90% 训练 / 10% 测试，论文 §3.4 协议）
输出：dict{data: (N, V, C, T), labels: (N,), dancer_ids: (N,), class_names}
"""
import os

import numpy as np

from .topology import JOINT_NAMES, N_JOINTS, base_pose, normalize_unit_cube

# 24 类运动基元定义（论文 Table 1 子类）
# 字段：主动关节组、运动模式、频率、幅度、主运动轴
CLASS_DEFS = [
    # 0 Lock-Point: 双腕高频点动（脉冲）
    {"joints": [4, 5, 12, 13], "mode": "pulse", "freq": 3.0, "amp": 0.045, "axis": 0},
    # 1 Lock-Scoop: 腕肘弧线舀动
    {"joints": [2, 3, 4, 5], "mode": "sin", "freq": 1.8, "amp": 0.07, "axis": 0},
    # 2 Lock-Wrist-Twirl: 腕小幅度高频旋转（易混淆：与 Waack-Arm-Swing）
    {"joints": [4, 5, 12, 13, 16, 17], "mode": "twirl", "freq": 3.5, "amp": 0.035, "axis": 1},
    # 3 Waack-Arm-Swing: 手臂大幅摆动
    {"joints": [0, 1, 2, 3, 4, 5, 12, 13], "mode": "sin", "freq": 2.2, "amp": 0.11, "axis": 1},
    # 4 Waack-Pose-Hold: 手臂定格（冻结 + 微颤）
    {"joints": [0, 1, 2, 3, 4, 5], "mode": "static", "freq": 0.4, "amp": 0.012, "axis": 0},
    # 5 Krump-Chest-Pop: 躯干脉冲
    {"joints": [21, 23, 22], "mode": "pulse", "freq": 2.4, "amp": 0.06, "axis": 1},
    # 6 Krump-Stomp: 足踩踏脉冲
    {"joints": [14, 15, 10, 11], "mode": "pulse", "freq": 1.6, "amp": 0.09, "axis": 1},
    # 7 Pop-Hit: 全身多关节短脉冲（律动）
    {"joints": list(range(24)), "mode": "pulse", "freq": 2.0, "amp": 0.04, "axis": 1},
    # 8 Pop-Stop: 全身急停（方波）
    {"joints": list(range(24)), "mode": "square", "freq": 1.2, "amp": 0.045, "axis": 1},
    # 9 Pop-Wave: 手臂波浪（相位沿链传播）
    {"joints": [0, 2, 4, 12, 16], "mode": "wave", "freq": 1.5, "amp": 0.08, "axis": 0},
    # 10 House-Footwork: 足快速步法
    {"joints": [10, 11, 14, 15, 8, 9], "mode": "sin", "freq": 3.2, "amp": 0.08, "axis": 0},
    # 11 House-Torso-Roll: 躯干滚动
    {"joints": [20, 21, 22, 23], "mode": "sin", "freq": 1.3, "amp": 0.055, "axis": 1},
    # 12 MidHip-Bounce: 髋部上下弹跳
    {"joints": [6, 7, 20, 21], "mode": "bounce", "freq": 2.6, "amp": 0.09, "axis": 1},
    # 13 MidHip-Glide: 髋部水平滑动（易混淆：与 Bounce）
    {"joints": [6, 7, 20, 21], "mode": "sin", "freq": 1.4, "amp": 0.085, "axis": 2},
    # 14 LAHip-Sway: 髋部左右摆动
    {"joints": [6, 7, 20], "mode": "sin", "freq": 1.0, "amp": 0.07, "axis": 0},
    # 15 LAHip-Drop: 髋部下沉
    {"joints": [6, 7, 8, 9, 20], "mode": "pulse", "freq": 1.1, "amp": 0.075, "axis": 1},
    # 16 LAHip-Freeze: 髋部冻结（低活动度）
    {"joints": [6, 7, 20], "mode": "static", "freq": 0.3, "amp": 0.008, "axis": 1},
    # 17 Break-Toprock: 站立步法（上下摇摆）
    {"joints": [0, 1, 2, 3, 4, 5, 6, 7], "mode": "sin", "freq": 1.9, "amp": 0.06, "axis": 1},
    # 18 Break-Footwork: 蹲姿地面步法
    {"joints": [6, 7, 8, 9, 10, 11, 14, 15], "mode": "square", "freq": 2.8, "amp": 0.07, "axis": 0},
    # 19 Break-Power: 腿部旋转踢
    {"joints": [6, 7, 8, 9, 10, 11, 14, 15], "mode": "twirl", "freq": 2.0, "amp": 0.1, "axis": 1},
    # 20 SJazz-Kick: 踝部踢（小幅度高频）
    {"joints": [10, 11, 14, 15], "mode": "pulse", "freq": 2.2, "amp": 0.085, "axis": 1},
    # 21 SJazz-Turn: 全身转身
    {"joints": list(range(24)), "mode": "sin", "freq": 0.8, "amp": 0.06, "axis": 2},
    # 22 BJazz-Extension: 臂腿伸展（方波延展）
    {"joints": [0, 1, 2, 3, 4, 5, 8, 9, 10, 11], "mode": "square", "freq": 1.0, "amp": 0.09, "axis": 0},
    # 23 BJazz-Pivot: 足转轴
    {"joints": [10, 11, 14, 15, 6, 7], "mode": "sin", "freq": 0.9, "amp": 0.07, "axis": 1},
]

CLASS_NAMES = [
    "Lock-Point", "Lock-Scoop", "Lock-Wrist-Twirl", "Waack-Arm-Swing",
    "Waack-Pose-Hold", "Krump-Chest-Pop", "Krump-Stomp", "Pop-Hit",
    "Pop-Stop", "Pop-Wave", "House-Footwork", "House-Torso-Roll",
    "MidHip-Bounce", "MidHip-Glide", "LAHip-Sway", "LAHip-Drop",
    "LAHip-Freeze", "Break-Toprock", "Break-Footwork", "Break-Power",
    "SJazz-Kick", "SJazz-Turn", "BJazz-Extension", "BJazz-Pivot",
]


def _motion_pattern(t, mode, freq, amp, axis_idx, phase):
    """t: (T,) 时间轴；返回 (T, 3) 偏移。"""
    T = len(t)
    if mode == "sin":
        s = amp * np.sin(2 * np.pi * freq * t + phase)
    elif mode == "pulse":
        # 周期窄高斯脉冲（Hit/Stomp/Point）
        period = 1.0 / freq
        phase_t = np.mod(t + phase / (2 * np.pi) * period, period)
        s = amp * np.exp(-((phase_t - period / 2) ** 2) / (2 * (period / 8) ** 2)) * 2
    elif mode == "square":
        s = amp * np.sign(np.sin(2 * np.pi * freq * t + phase))
    elif mode == "twirl":
        s = amp * np.sin(2 * np.pi * freq * t + phase) * np.sin(np.pi * t * 2)
    elif mode == "bounce":
        s = amp * np.abs(np.sin(2 * np.pi * freq * t + phase))
    elif mode == "wave":
        # 沿关节链传播的相位波（手臂波浪）
        s = amp * np.sin(2 * np.pi * freq * t + phase)
    elif mode == "static":
        s = amp * np.sin(2 * np.pi * freq * t + phase)
    else:
        s = np.zeros(T)
    out = np.zeros((T, 3), dtype=np.float32)
    out[:, axis_idx] = s
    if mode == "wave":
        out[:, 0] = s
        out[:, 1] = 0.4 * amp * np.cos(2 * np.pi * freq * t + phase)
    if mode == "twirl":
        # 旋转：两个轴上的正交分量
        out[:, axis_idx] = amp * np.sin(2 * np.pi * freq * t + phase)
        out[:, (axis_idx + 1) % 3] = 0.7 * amp * np.cos(2 * np.pi * freq * t + phase)
    return out


def _chain_wave(t, chain, freq, amp, phase, axis):
    """沿关节链传播的相位波（Pop-Wave / 关节链波浪）。"""
    T = len(t)
    offsets = np.zeros((T, len(chain), 3), dtype=np.float32)
    for k, j in enumerate(chain):
        d = amp * np.sin(2 * np.pi * freq * t + phase + k * 0.9)
        offsets[:, k, axis] = d
    return offsets


def generate_sample(cls_id, t, dancer_style, noise=0.015, beat_jitter=0.25, rng=None):
    """生成单个样本 (T, V, C)。dancer_style: 该舞者的频率/幅度风格偏移。"""
    if rng is None:
        rng = np.random.default_rng()
    d = CLASS_DEFS[cls_id]
    base = base_pose()                                   # (V, C)
    # 个体节拍偏移（式7/8 要消除的相位漂移）
    phase = rng.uniform(0, 2 * np.pi) * (1 + beat_jitter * rng.standard_normal() * 0)
    # 频率/幅度个体差异
    f = d["freq"] * (1 + dancer_style * 0.15 + rng.uniform(-0.1, 0.1))
    amp = d["amp"] * (1 + rng.uniform(-0.15, 0.15))

    seq = np.tile(base[None, :, :], (len(t), 1, 1)).astype(np.float32)  # (T, V, C)
    joints = d["joints"]
    if d["mode"] == "wave" and d["freq"] == 1.5:
        off = _chain_wave(t, joints, f, amp * 1.2, phase, d["axis"])
        for k, j in enumerate(joints):
            seq[:, j] += off[:, k]
    else:
        off = _motion_pattern(t, d["mode"], f, amp, d["axis"], phase)
        for j in joints:
            seq[:, j] += off

    # 全身轻微呼吸运动（低幅）
    breath = 0.004 * np.sin(2 * np.pi * 0.5 * t + phase)
    seq[:, :, 1] += breath[:, None]

    # 传感器/姿态估计噪声
    seq += rng.normal(0, noise, seq.shape).astype(np.float32)

    # 归一化到单位立方体
    seq = normalize_unit_cube(seq)
    return seq.astype(np.float32)                        # (T, V, C)


def generate_dataset(cfg, cache_dir="data_cache"):
    """生成并缓存合成数据集。返回 dict(data, labels, dancer_ids, class_names)。"""
    s = cfg["data"]["synth"]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"synth_dance_{s['n_frames']}f_{s['samples_per_class']}spc.npz")
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=True)
        return {"data": z["data"], "labels": z["labels"], "dancer_ids": z["dancer_ids"],
                "class_names": z["class_names"].tolist()}

    rng = np.random.default_rng(s["seed"])
    T = s["n_frames"]
    t = np.arange(T) / 60.0                              # 60Hz
    n_dancers = s["n_dancers"]
    dancer_style = rng.normal(0, 1, n_dancers)           # 舞者个体风格（频率漂移）

    n_per_class = s["samples_per_class"]
    N = len(CLASS_NAMES) * n_per_class
    data = np.zeros((N, T, N_JOINTS, 3), dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    dancer_ids = np.zeros(N, dtype=np.int64)

    idx = 0
    for c in range(len(CLASS_NAMES)):
        for k in range(n_per_class):
            dancer = rng.integers(0, n_dancers)
            data[idx] = generate_sample(c, t, dancer_style[dancer],
                                        noise=s["noise"], beat_jitter=s["beat_jitter"], rng=rng)
            labels[idx] = c
            dancer_ids[idx] = dancer
            idx += 1

    # 布局转 (N, V, C, T)（模型输入：关节 × 通道 × 时间）
    data = np.transpose(data, (0, 2, 3, 1)).astype(np.float32)
    out = {"data": data, "labels": labels, "dancer_ids": dancer_ids,
           "class_names": np.array(CLASS_NAMES)}
    np.savez_compressed(cache_path, **out)
    return out
