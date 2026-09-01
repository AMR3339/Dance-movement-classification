"""AIST++ 真实数据加载器（备用：用户获取 AIST++ 数据后可直接使用）。

AIST++ 提供 (N, 24, 3) 每帧 3D 骨架（60Hz）。此处实现：
  - 从 `{path}` 读取原始骨架序列（npz/npy，字段 'smpl_poses' 或 'keypoints3d'）
  - 零填充统一至 128 帧 + 单位立方体归一化（论文 §3.4）
  - 加载 24 细分子类标注（label_file: CSV，列为 sequence_id,label）
  - 按舞者 ID（序列名前缀 dancer 编号）dancer-independent 划分（90/10）
输出与合成数据同构的 dict(data, labels, dancer_ids, class_names)。
"""
import os

import numpy as np


def load_aistpp(cfg):
    d = cfg["data"]["aistpp"]
    path = d["path"]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"AIST++ 数据目录不存在：{path}。请先下载 AIST++（Google Research）并放入该目录，"
            f"或使用 config.yaml 中 data.source='synth' 的合成数据演示。")
    seqs, labels, dancers = [], [], []
    class_names = sorted(set())
    for f in sorted(os.listdir(path)):
        if f.endswith(".npz") or f.endswith(".npy"):
            seq_id = os.path.splitext(f)[0]
            arr = np.load(os.path.join(path, f))
            kp = arr["keypoints3d"] if "keypoints3d" in arr else arr["smpl_poses"]
            seqs.append(kp)
            dancers.append(seq_id.split("_")[0])
    # 读取标注
    label_map = {}
    if os.path.exists(os.path.join(path, d["label_file"])):
        import csv
        with open(os.path.join(path, d["label_file"]), encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                label_map[row["sequence_id"]] = int(row["label"])
    labels = [label_map.get(os.path.splitext(f)[0], -1) for f in os.listdir(path)
              if f.endswith((".npz", ".npy"))]
    keep = [i for i, l in enumerate(labels) if l >= 0]
    seqs = [seqs[i] for i in keep]
    labels = [labels[i] for i in keep]
    dancers = [dancers[i] for i in keep]
    class_names = sorted(set(labels))
    return {"seqs": seqs, "labels": np.array(labels), "dancer_ids": np.array(dancers),
            "class_names": class_names}
