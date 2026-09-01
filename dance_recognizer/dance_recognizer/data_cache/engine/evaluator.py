"""评估器（论文 §3.4/4.x 五维指标体系）。

- Top-1 Accuracy、Macro F1、Cross-Pair F1（易混淆子类对）
- FDI（式12，Davies-Bouldin 风格）：FDI = (1/K) sum_i max_j (S_i+S_j)/d_ij
- 聚类质量：平均轮廓系数 / Davies-Bouldin 指数 / 平均类内距离（论文表5）
- 混淆矩阵（图3）、类别难度系数 = 1 - per-class precision（图4）
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (confusion_matrix, silhouette_score, davies_bouldin_score,
                             f1_score)


def macro_f1(cm):
    """由混淆矩阵计算 Macro F1。"""
    K = cm.shape[0]
    prec = np.zeros(K)
    rec = np.zeros(K)
    for i in range(K):
        prec[i] = cm[i, i] / max(cm[:, i].sum(), 1e-9)
        rec[i] = cm[i, i] / max(cm[i, :].sum(), 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return float(f1.mean())


def cross_pair_f1(labels, preds, pairs):
    """易混淆子类对 F1：对每对 (a,b)，在 a∪b 样本子集上分别计算两类的二分类 F1 后平均。"""
    scores = []
    for (a, b) in pairs:
        mask = np.isin(labels, [a, b])
        if mask.sum() == 0:
            continue
        y_true, y_pred = labels[mask], preds[mask]
        f1_a = f1_score((y_true == a).astype(int), (y_pred == a).astype(int),
                        average="binary", zero_division=0)
        f1_b = f1_score((y_true == b).astype(int), (y_pred == b).astype(int),
                        average="binary", zero_division=0)
        scores.append((f1_a + f1_b) / 2)
    return float(np.mean(scores)) if scores else 0.0


def fdi_index(emb, labels, K):
    """式(12)：FDI —— Davies-Bouldin 风格类内紧凑/类间分离比。"""
    centroids = np.stack([emb[labels == c].mean(axis=0) for c in range(K)])
    S = np.zeros(K)
    for c in range(K):
        pts = emb[labels == c]
        if len(pts) == 0:
            continue
        S[c] = np.mean(np.linalg.norm(pts - centroids[c], axis=1))
    D = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
    D[D < 1e-9] = 1e9
    fdi = 0.0
    for i in range(K):
        fdi += np.max((S[i] + S) / D[i]) if np.isfinite(np.max((S[i] + S) / D[i])) else 0.0
    return float(fdi / K)


def cluster_metrics(emb, labels):
    """平均轮廓系数 / Davies-Bouldin / 平均类内距离（论文表5）。"""
    if len(set(labels.tolist())) < 2 or len(emb) < 3:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"),
                "intra_class_dist": float("nan")}
    sil = float(silhouette_score(emb, labels))
    db = float(davies_bouldin_score(emb, labels))
    K = len(set(labels.tolist()))
    dists = []
    for c in range(K):
        pts = emb[labels == c]
        if len(pts) > 1:
            cent = pts.mean(axis=0)
            dists.append(np.mean(np.linalg.norm(pts - cent, axis=1)))
    return {"silhouette": sil, "davies_bouldin": db,
            "intra_class_dist": float(np.mean(dists)) if dists else float("nan")}


@torch.no_grad()
def evaluate(model, loader, prototypes, device, cross_pairs=None, tau_metric=0.2):
    """完整评估。返回指标 dict + (cm, emb, labels, logits)。"""
    model.eval()
    all_logits, all_emb, all_labels = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits_metric, emb, _ = model(x, prototypes=prototypes, tau_metric=tau_metric)
        if logits_metric is None:
            logits_metric = torch.zeros(x.shape[0], model.num_classes, device=device)
        all_logits.append(logits_metric.cpu())
        all_emb.append(emb.cpu())
        all_labels.append(y.cpu())
    logits = torch.cat(all_logits).numpy()
    emb = torch.cat(all_emb).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = logits.argmax(1)
    K = model.num_classes
    cm = confusion_matrix(labels, preds, labels=list(range(K)))

    top1 = float((preds == labels).mean())
    mf1 = macro_f1(cm)
    cf1 = cross_pair_f1(labels, preds, cross_pairs or [])
    fdi = fdi_index(emb, labels, K)
    clust = cluster_metrics(emb, labels)
    per_class_precision = cm.diagonal() / np.maximum(cm.sum(axis=0), 1e-9)
    difficulty = 1.0 - per_class_precision
    return {"top1": top1, "macro_f1": mf1, "cross_pair_f1": cf1, "fdi": fdi,
            "n_test": int(len(labels))}, {"cm": cm, "emb": emb, "labels": labels,
                                          "difficulty": difficulty, "logits": logits}
