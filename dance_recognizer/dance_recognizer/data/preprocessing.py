"""预处理管线（论文 §3.2）：离散曲率算子增强 + 节拍对齐。

流程：
  1. 一维高斯滤波抑制传感器/姿态估计高频抖动；
  2. 质心对齐 + 单位立方体归一化（消除舞者身高与拍摄角度差异）；
  3. 轨迹离散曲率增强（式5 曲率算子，式6 曲率驱动增强映射）：
       x' = x + beta · Sigmoid((kappa - tau)/kappa0) · (x[t+1]-x[t-1])/2
     沿轨迹切线方向施加与曲率正相关的位移补偿，放大末端关节（腕/踝）的高频微变形；
  4. 节拍对齐（式7 运动能量包络，式8 相位差动态时间重映射）：
     基于训练集离线构建全局标准节拍相位模板（固定，测试期无循环依赖），
     通过互相关求个体序列的瞬时相位偏差，三次样条插值重投影到标准节拍网格。
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline


def gaussian_filter_seq(seq, sigma=1.5):
    """对 (N, V, C, T) 或 (V, C, T) 沿时间维高斯滤波。"""
    axis = -1 if seq.ndim == 3 else -1
    return gaussian_filter1d(seq, sigma=sigma, axis=axis)


def _curvature(seq):
    """式(5)：离散曲率算子。seq: (T, V, C) -> (T, V) 曲率标量。

    v = 中心差分之一阶速度，a = 二阶加速度
    kappa = ||v x a|| / (||v||^3 + eps)
    """
    v = np.zeros_like(seq)
    v[1:-1] = (seq[2:] - seq[:-2]) / 2.0
    a = np.zeros_like(seq)
    a[1:-1] = seq[2:] - 2 * seq[1:-1] + seq[:-2]
    vn = np.linalg.norm(v, axis=-1) + 1e-8
    cross = np.cross(v, a)
    kappa = np.linalg.norm(cross, axis=-1) / (vn ** 3 + 1e-8)
    return kappa


def curvature_enhance(seq, beta=0.1, tau=0.03, kappa0=0.01):
    """式(6)：曲率驱动增强。seq: (T, V, C) -> (T, V, C)。

    x' = x + beta * sigmoid((kappa - tau)/kappa0) * v_hat
    v_hat 为归一化切线方向（中心差分之一阶速度方向）。
    """
    v = np.zeros_like(seq)
    v[1:-1] = (seq[2:] - seq[:-2]) / 2.0
    kappa = _curvature(seq)
    gate = 1.0 / (1.0 + np.exp(-(kappa - tau) / kappa0))     # Sigmoid 激活阈值
    vn = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8
    tangent = v / vn                                          # 单位切向
    enhanced = seq + beta * gate[..., None] * tangent
    return enhanced.astype(np.float32)


def motion_energy(seq):
    """式(7)：帧级运动能量包络 E(t) = sum_j ||v_j(t)||^2。seq: (T, V, C) -> (T,)"""
    v = np.zeros_like(seq)
    v[1:-1] = (seq[2:] - seq[:-2]) / 2.0
    return np.sum(np.linalg.norm(v, axis=-1) ** 2, axis=-1)


def build_beat_template(train_energy):
    """离线构建全局标准节拍相位模板（训练集统计平均）。

    返回 (template, period)：模板为平均能量包络；period 为主导节奏周期（帧）。
    """
    L = min(len(e) for e in train_energy)
    env = np.stack([e[:L] for e in train_energy]).mean(axis=0)
    env = gaussian_filter1d(env, sigma=2.0)
    # 自相关提取主导周期
    ac = np.correlate(env - env.mean(), env - env.mean(), mode="full")[len(env) - 1:]
    candidates = np.arange(8, len(env) // 3)
    if len(candidates) == 0:
        period = 32
    else:
        period = int(candidates[np.argmax(ac[candidates])])
    return env, period


def beat_align(seq, template, period, max_shift=40):
    """式(8)：相位偏差重映射 + 三次样条子帧插值对齐。

    通过样本能量包络与全局模板的互相关求相位偏差 delta（限制在 ±period/2 的
    相位范围内，消除节拍周期模糊，对齐到标准节拍相位），
    时间重映射 t' = t - delta，CubicSpline 重采样到标准节拍网格。
    seq: (T, V, C) -> (T, V, C)
    """
    T, V, C = seq.shape
    env = motion_energy(seq)
    env = gaussian_filter1d(env, sigma=2.0)
    tl = min(T, len(template))
    a = env[:tl] - env[:tl].mean()
    b = template[:tl] - template[:tl].mean()
    # 相位对齐：搜索范围限制在 ±period/2（避免周期模糊）
    max_shift = min(max_shift, max(period // 2, 4))
    # 全互相关，中心偏移 T-1；限制在 ±max_shift 内
    full = np.correlate(a, b, mode="full")
    center = len(full) // 2
    lo, hi = max(0, center - max_shift), min(len(full), center + max_shift + 1)
    best_delta = int(np.argmax(full[lo:hi]) + lo - center)
    # 时间重映射 t' = t - delta（子帧级，三次样条）
    t_old = np.arange(T, dtype=np.float64)
    t_new = np.clip(t_old - best_delta, 0, T - 1)
    out = np.zeros_like(seq)
    for v in range(V):
        for c in range(C):
            cs = CubicSpline(t_old, seq[:, v, c])
            out[:, v, c] = cs(t_new)
    return out.astype(np.float32), best_delta


def preprocess_pipeline(data, sigma=1.5, beta=0.1, tau=0.03, kappa0=0.01,
                        align=True, template=None, period=None):
    """完整预处理：data (N, V, C, T) -> (N, V, C, T)（曲率增强 + 可选节拍对齐）。

    返回 (processed, meta)，meta 含每样本对齐位移（供可视化）。
    """
    N, V, C, T = data.shape
    data_tvc = np.transpose(data, (0, 3, 1, 2))            # (N, T, V, C)
    # 1. 高斯滤波
    for i in range(N):
        data_tvc[i] = gaussian_filter_seq(data_tvc[i], sigma)
    # 2. 质心/尺度归一化（单位立方体）
    for i in range(N):
        data_tvc[i] = normalize_tvc(data_tvc[i])
    # 3. 曲率增强
    for i in range(N):
        data_tvc[i] = curvature_enhance(data_tvc[i], beta, tau, kappa0)
    shifts = np.zeros(N, dtype=np.float32)
    if align and template is not None:
        for i in range(N):
            data_tvc[i], shifts[i] = beat_align(data_tvc[i], template, period)
    return np.transpose(data_tvc, (0, 2, 3, 1)).astype(np.float32), shifts


def normalize_tvc(seq):
    """(T, V, C) 质心对齐 + 单位立方体归一化。"""
    cent = seq.mean(axis=(0, 1), keepdims=True)
    seq = seq - cent
    span = seq.max() - seq.min() + 1e-8
    return seq / span
