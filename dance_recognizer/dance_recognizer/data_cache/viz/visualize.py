"""各阶段中间结果可视化（对应论文图3-图8）。

stage0 数据统计、stage1 预处理轨迹/曲率/能量包络、stage2 自适应邻接矩阵、
stage3 训练曲线 + GSNR + 原型 PCA、stage4 混淆矩阵 + 类别难度、stage5 嵌入流形。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


def _trajectory(seq, joint):
    """取关节轨迹 (V,C,T) -> (T, 3)。"""
    return seq[joint].T


# ---------------- Stage 0 ----------------
def stage0_class_distribution(labels, class_names, out_dir):
    counts = np.bincount(labels, minlength=len(class_names))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(class_names)), counts, color="#4C72B0")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("样本数")
    ax.set_title(f"24 细粒度子类样本分布（共 {len(labels)} 条）")
    _save(fig, os.path.join(out_dir, "class_distribution.png"))


def stage0_trajectory_sample(seq, joints, out_dir):
    """骨架序列轨迹样例：腕/踝关节 3D 轨迹（论文图7/8 的原始形态）。seq: (T, V, C)"""
    fig = plt.figure(figsize=(10, 4))
    for k, j in enumerate(joints):
        ax = fig.add_subplot(1, len(joints), k + 1, projection="3d")
        tr = _trajectory(seq, j)
        ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], lw=1.2)
        ax.scatter(tr[0, 0], tr[0, 1], tr[0, 2], c="red", s=14, label="start")
        ax.set_title(f"joint {j}")
        ax.set_xlabel("x")
    fig.suptitle("关节轨迹样例（原始）")
    _save(fig, os.path.join(out_dir, "trajectory_samples.png"))


# ---------------- Stage 1 预处理 ----------------
def stage1_preprocess_compare(raw_tvc, curv_tvc, aligned_tvc, template_env, shifts,
                              class_names, cls_id, joints, out_dir):
    """预处理对比（论文图7/8）：原始 vs 曲率增强 vs 节拍对齐的末端关节轨迹 + 曲率分布。"""
    fig = plt.figure(figsize=(16, 8))
    titles = ["原始轨迹", "曲率增强后（式6）", "节拍对齐后（式8）"]
    seqs = [raw_tvc, curv_tvc, aligned_tvc]
    for k, seq in enumerate(seqs):
        ax = fig.add_subplot(2, 3, k + 1, projection="3d")
        for j in joints:
            tr = _trajectory(seq, j)
            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], lw=1.0)
        ax.set_title(f"{titles[k]}（关节 {joints}）")
        ax.set_xlabel("x")
    # 曲率分布对比
    from data.preprocessing import _curvature
    ax2 = fig.add_subplot(2, 3, 4)
    for k, (seq, lab) in enumerate(zip(seqs, titles)):
        curv = _curvature(seq)
        ax2.hist(curv.flatten(), bins=40, alpha=0.55, label=lab)
    ax2.set_yscale("log")
    ax2.set_xlabel("曲率 κ")
    ax2.set_title("曲率分布（高曲率比例：增强后提升）")
    ax2.legend(fontsize=8)
    # 能量包络对齐
    from data.preprocessing import motion_energy
    ax3 = fig.add_subplot(2, 3, 5)
    e0 = motion_energy(raw_tvc)
    e1 = motion_energy(aligned_tvc)
    ax3.plot(e0 / (e0.max() + 1e-8), alpha=0.7, label="原始能量包络")
    ax3.plot(e1 / (e1.max() + 1e-8), alpha=0.7, label="对齐后能量包络")
    if template_env is not None:
        ax3.plot(template_env / (template_env.max() + 1e-8), "k--", label="全局节拍模板")
    ax3.set_xlabel("帧")
    ax3.set_title("运动能量包络（式7）与节拍对齐")
    ax3.legend(fontsize=8)
    # 对齐位移分布
    ax4 = fig.add_subplot(2, 3, 6)
    ax4.hist(shifts, bins=30, color="#C44E52")
    ax4.set_xlabel("相位位移 δ（帧）")
    ax4.set_title(f"节拍对齐位移分布 (n={len(shifts)})")
    fig.suptitle(f"预处理效果可视化：{class_names[cls_id]}")
    _save(fig, os.path.join(out_dir, "preprocess_compare.png"))


def stage1_template(template_env, period, out_dir):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(template_env)
    ax.axvspan(0, period, alpha=0.15, color="green", label=f"主导周期 {period} 帧")
    ax.set_xlabel("帧")
    ax.set_title("全局标准节拍相位模板（训练集离线统计）")
    ax.legend()
    _save(fig, os.path.join(out_dir, "beat_template.png"))


# ---------------- Stage 2 邻接矩阵 ----------------
def stage2_adjacency(adapt_mat, A_phys, out_dir):
    """式(2)(3)(4) 自适应邻接矩阵 vs 物理邻接。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    A = adapt_mat if adapt_mat.ndim == 2 else adapt_mat[0]
    im0 = axes[0].imshow(A_phys[0], cmap="Blues")
    axes[0].set_title("物理 1-hop 邻接")
    axes[1].imshow(A_phys[1], cmap="Blues")
    axes[1].set_title("2-hop 跨肢邻接")
    im2 = axes[2].imshow(A, cmap="viridis")
    axes[2].set_title("自适应邻接 A_adapt（式4）")
    for ax in axes:
        ax.set_xlabel("关节")
        ax.set_ylabel("关节")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)
    _save(fig, os.path.join(out_dir, "adaptive_adjacency.png"))


# ---------------- Stage 3 训练 ----------------
def stage3_curves(history, out_dir):
    """损失演化 + GSNR 双轴（论文图5）。"""
    epochs = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, [h["train_loss"] for h in history], label="train loss", color="#4C72B0")
    ax1.plot(epochs, [h["ce"] for h in history], label="weighted CE", color="#55A868")
    ax1.plot(epochs, [h["proto"] for h in history], label="prototype contrast", color="#8172B3")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc="upper right")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [h["gsnr"] for h in history], "o-", color="#C44E52", ms=3, label="GSNR")
    ax2.set_ylabel("GSNR（梯度信噪比）", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    ax2.set_ylim(0, max([h["gsnr"] for h in history], default=1) * 1.2)
    ax1.set_title("联合损失收敛 + 梯度信噪比演化（论文图5）")
    _save(fig, os.path.join(out_dir, "training_gsnr.png"))

    fig2, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, [h["lr"] for h in history], color="#C44E52", marker="o", ms=3)
    ax.set_xlabel("epoch")
    ax.set_ylabel("learning rate")
    ax.set_title("两阶段学习率调度（论文 §3.4）")
    _save(fig2, os.path.join(out_dir, "lr_schedule.png"))


def stage3_prototype_pca(proto_history, class_names, out_dir):
    """24 类原型在 3D PCA 空间的收敛轨迹（论文图6）。"""
    H, K, D = proto_history.shape
    flat = proto_history.reshape(-1, D)
    mu = flat.mean(axis=0)
    _, _, Vt = np.linalg.svd(flat - mu, full_matrices=False)
    top3 = Vt[:3].T                                   # (D, 3) 主方向
    proj = (flat - mu) @ top3
    proj = proj.reshape(H, K, 3)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab20(np.linspace(0, 1, K))
    for k in range(K):
        ax.plot(proj[:, k, 0], proj[:, k, 1], proj[:, k, 2], lw=0.9, color=colors[k])
        ax.scatter(proj[0, k, 0], proj[0, k, 1], proj[0, k, 2], color=colors[k], s=12)
        ax.scatter(proj[-1, k, 0], proj[-1, k, 1], proj[-1, k, 2], color=colors[k], s=22)
    ax.set_title("类原型收敛轨迹（3D PCA，论文图6）")
    _save(fig, os.path.join(out_dir, "prototype_pca.png"))


# ---------------- Stage 4 评估 ----------------
def stage4_confusion(cm, class_names, out_dir):
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=6)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    fig.colorbar(im, fraction=0.046)
    ax.set_title("24 类混淆矩阵（论文图3）")
    _save(fig, os.path.join(out_dir, "confusion_matrix.png"))


def stage4_difficulty(difficulty, class_names, out_dir):
    fig, ax = plt.subplots(figsize=(12, 4))
    order = np.argsort(-difficulty)
    ax.bar(range(len(class_names)), difficulty[order], color="#DD8452")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels([class_names[i] for i in order], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("难度系数 (1 - 类别精度)")
    ax.set_title("类别难度分布（论文图4）")
    _save(fig, os.path.join(out_dir, "category_difficulty.png"))


# ---------------- Stage 5 嵌入流形 ----------------
def stage5_tsne(emb, labels, class_names, out_dir):
    from sklearn.manifold import TSNE
    if len(emb) < 5:
        return
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, len(emb) // 5)))
    e2 = tsne.fit_transform(emb)
    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))
    for c in range(len(class_names)):
        sel = labels == c
        if sel.sum():
            ax.scatter(e2[sel, 0], e2[sel, 1], c=[colors[c]], s=14, label=class_names[c], alpha=0.75)
    ax.set_title("测试集 256 维嵌入 t-SNE 流形")
    ax.legend(fontsize=6, ncol=2, loc="upper left", markerscale=0.6)
    _save(fig, os.path.join(out_dir, "tsne_embedding.png"))
