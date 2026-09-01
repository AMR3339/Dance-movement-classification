"""数据集封装：dancer-independent 划分 + 训练/测试 DataLoader。"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SkeletonDataset(Dataset):
    """骨架序列数据集。data: (N, V, C, T)。"""

    def __init__(self, data, labels, augment=False, seed=42):
        self.data = torch.from_numpy(np.ascontiguousarray(data)).float()
        self.labels = torch.from_numpy(np.asarray(labels)).long()
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        if self.augment:
            # 时间维随机裁剪/抖动 + 高斯噪声（轻量增强，保持时序结构）
            if self.rng.random() < 0.5:
                shift = self.rng.integers(-4, 5)
                x = torch.roll(x, int(shift), dims=-1)
            if self.rng.random() < 0.3:
                x = x + torch.randn_like(x) * 0.005
        return x, self.labels[idx]


def split_by_dancer(data, labels, dancer_ids, train_ratio=0.9, seed=42):
    """按舞者 ID 划分（论文 §3.4 dancer-independent 协议），组内分层抽样。"""
    rng = np.random.default_rng(seed)
    dancers = np.unique(dancer_ids)
    rng.shuffle(dancers)
    n_train = max(1, int(len(dancers) * train_ratio))
    train_dancers = set(dancers[:n_train].tolist())
    tr_mask = np.array([d in train_dancers for d in dancer_ids])
    # 组内分层：保证每类在训练/测试都有样本
    train_idx, test_idx = [], []
    for c in np.unique(labels):
        c_mask = (labels == c) & tr_mask
        c_test = (labels == c) & ~tr_mask
        train_idx.extend(np.where(c_mask)[0].tolist())
        test_idx.extend(np.where(c_test)[0].tolist())
    train_idx = np.array(sorted(train_idx))
    test_idx = np.array(sorted(test_idx))
    if len(test_idx) == 0:
        # 兜底：随机留出 10% 样本
        all_idx = np.arange(len(labels))
        rng.shuffle(all_idx)
        n_test = max(1, int(len(labels) * (1 - train_ratio)))
        test_idx = all_idx[:n_test]
        train_idx = np.setdiff1d(all_idx, test_idx)
    return train_idx, test_idx


def make_loaders(data, labels, dancer_ids, cfg, batch_size=32):
    train_idx, test_idx = split_by_dancer(data, labels, dancer_ids,
                                          train_ratio=cfg["data"]["synth"]["train_ratio"],
                                          seed=cfg["train"]["seed"])
    train_ds = SkeletonDataset(data[train_idx], labels[train_idx], augment=True)
    test_ds = SkeletonDataset(data[test_idx], labels[test_idx], augment=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader, train_idx, test_idx
