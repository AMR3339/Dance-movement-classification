"""24 关节人体骨架拓扑（与论文 Table 6 / AIST++ 一致）。

关节顺序（0-23）：
  0 左肩  1 右肩  2 左肘  3 右肘  4 左腕  5 右腕
  6 左髋  7 右髋  8 左膝  9 右膝  10 左踝 11 右踝
  12 左手 13 右手 14 左足 15 右足 16 左手尖 17 右手尖
  18 左拇指 19 右拇指 20 脊柱底 21 脊柱中 22 颈 23 脊柱上
"""
import numpy as np

JOINT_NAMES = [
    "LeftShoulder", "RightShoulder", "LeftElbow", "RightElbow",
    "LeftWrist", "RightWrist", "LeftHip", "RightHip",
    "LeftKnee", "RightKnee", "LeftAnkle", "RightAnkle",
    "LeftHand", "RightHand", "LeftFoot", "RightFoot",
    "LeftHandTip", "RightHandTip", "LeftThumb", "RightThumb",
    "BaseSpine", "MidSpine", "Neck", "SpineUpper",
]

N_JOINTS = 24

# 物理骨架边（骨骼连接）：脊柱链 + 四肢链
PHYSICAL_EDGES = [
    (20, 21), (21, 23), (23, 22),          # 脊柱底-中-上-颈
    (0, 22), (1, 22),                       # 肩-颈
    (0, 23), (1, 23),                       # 肩-脊柱上
    (0, 2), (2, 4), (4, 12), (12, 16), (4, 18),   # 左臂链（肩-肘-腕-手-手尖/拇指）
    (1, 3), (3, 5), (5, 13), (13, 17), (5, 19),   # 右臂链
    (6, 8), (8, 10), (10, 14),              # 左腿链（髋-膝-踝-足）
    (7, 9), (9, 11), (11, 15),              # 右腿链
    (6, 20), (7, 20),                       # 髋-脊柱底
]

# 跳跃子图（跨肢/对角连接，模拟跳跃相关动力学）
JUMP_EDGES = [
    (0, 1), (6, 7), (4, 5), (10, 11),       # 左右对称镜像
    (0, 6), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11),  # 上肢-下肢对角
]


def build_adjacency(n_joints=N_JOINTS):
    """多尺度邻接矩阵：scale0 = 1-hop 物理边，scale1 = 2-hop 跨肢，scale2 = 跳跃子图。"""
    A1 = np.zeros((n_joints, n_joints), dtype=np.float32)
    for (i, j) in PHYSICAL_EDGES:
        A1[i, j] = A1[j, i] = 1.0
    A2 = (A1 @ A1 > 0).astype(np.float32)   # 2-hop
    np.fill_diagonal(A2, 0.0)
    A2 = np.clip(A2 - A1, 0, 1)             # 去掉 1-hop 重叠
    A3 = np.zeros((n_joints, n_joints), dtype=np.float32)
    for (i, j) in JUMP_EDGES:
        A3[i, j] = A3[j, i] = 1.0
    A3 = np.clip(A3 - A1 - A2, 0, 1)
    return [A1, A2, A3]


def normalized_adjacency(A):
    """D^-0.5 A D^-0.5 对称归一化（式1 图聚合用）。"""
    deg = A.sum(axis=1) + 1e-10
    d = np.diag(1.0 / np.sqrt(deg))
    return d @ A @ d


# ---------------- 静态基础姿态（单位立方体，y 向上） ----------------
def base_pose():
    """站立基础骨架 (24, 3)。"""
    pose = np.array([
        [-0.16, 1.05, 0.00], [-0.16 + 0.32, 1.05, 0.00],  # 0/1 肩
        [-0.24, 0.85, -0.05], [0.24, 0.85, -0.05],        # 2/3 肘
        [-0.30, 0.68, -0.10], [0.30, 0.68, -0.10],        # 4/5 腕
        [-0.11, 0.55, 0.00], [0.11, 0.55, 0.00],          # 6/7 髋
        [-0.13, 0.28, 0.00], [0.13, 0.28, 0.00],          # 8/9 膝
        [-0.14, 0.02, 0.00], [0.14, 0.02, 0.00],          # 10/11 踝
        [-0.31, 0.65, -0.12], [0.31, 0.65, -0.12],        # 12/13 手
        [-0.15, 0.00, 0.05], [0.15, 0.00, 0.05],          # 14/15 足
        [-0.33, 0.62, -0.13], [0.33, 0.62, -0.13],        # 16/17 手尖
        [-0.31, 0.66, -0.08], [0.31, 0.66, -0.08],        # 18/19 拇指
        [0.00, 0.55, 0.00], [0.00, 0.80, 0.00],           # 20/21 脊柱底/中
        [0.00, 1.10, 0.00], [0.00, 0.98, 0.00],           # 22/23 颈/脊柱上
    ], dtype=np.float32)
    return pose


def normalize_unit_cube(seq):
    """质心对齐 + 缩放至单位立方体（文章 §3.2 尺度归一化）。seq: (T, V, C) 或 (V, C)"""
    centroid = seq.mean(axis=tuple(range(seq.ndim - 1)), keepdims=True)
    seq = seq - centroid
    span = seq.max() - seq.min() + 1e-8
    return seq / span
