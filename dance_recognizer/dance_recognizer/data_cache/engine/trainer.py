"""训练器：AdamW + 两阶段学习率调度 + 梯度信噪比(GSNR) 记录 + 原型动量更新。

- 优化器 AdamW（lr/weight_decay 网格中选取，论文 §3.4）；
- 学习率：warmup 后按 lr_decay_epochs 两阶段衰减（论文图5 阶段切换）；
- 原型动量更新（式9）逐 batch 执行；
- GSNR = |mean(grad)| / (std(grad)+eps)，逐 iter 记录，验证优化稳定性（论文图5）。
"""
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from models.losses import PrototypeMemory, compute_loss, class_weights


class Trainer:
    def __init__(self, model, cfg, device, train_labels=None):
        self.model = model
        self.cfg = cfg
        self.device = device
        t, l = cfg["train"], cfg["loss"]
        self.epochs = t["epochs"]
        self.lr = t["lr"]
        self.weight_decay = t["weight_decay"]
        self.lr_decay_epochs = t["lr_decay_epochs"]
        self.warmup_epochs = t["warmup_epochs"]
        self.tau_metric = l["proto_temp"]
        self.proto_temp = l["proto_temp"]
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                           weight_decay=self.weight_decay)
        self.proto_mem = PrototypeMemory(model.num_classes, model.emb_dim,
                                         momentum=l["proto_momentum"], device=device)
        self.ce_weights = None
        if train_labels is not None:
            self.ce_weights = class_weights(torch.as_tensor(train_labels), model.num_classes).to(device)

    def _lr_at_epoch(self, epoch):
        """warmup + 两阶段衰减。"""
        if epoch < self.warmup_epochs:
            return self.lr * (epoch + 1) / max(self.warmup_epochs, 1)
        lr = self.lr
        for d in self.lr_decay_epochs:
            if epoch >= d:
                lr *= 0.1
        return lr

    def train_one_epoch(self, loader, epoch):
        self.model.train()
        tot, n_iter = 0.0, 0
        comps = {"ce": 0.0, "proto": 0.0, "reg": 0.0}
        gsnrs = []
        lr = self._lr_at_epoch(epoch)
        for g in self.optimizer.param_groups:
            g["lr"] = lr

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            protos = self.proto_mem.get()
            logits_metric, emb, adapt_mats = self.model(x, prototypes=protos,
                                                        tau_metric=self.tau_metric)
            if logits_metric is None:   # 首轮原型未建立，退化用随机温度 logits
                logits_metric = torch.randn(x.shape[0], self.model.num_classes,
                                            device=self.device) * 0.1
            total, comp = compute_loss(logits_metric, y, emb, protos, adapt_mats,
                                       self.cfg["loss"], ce_weights=self.ce_weights)
            self.optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            # GSNR（梯度信号质量）
            grads = [p.grad.flatten() for p in self.model.parameters()
                     if p.grad is not None and p.grad.numel() > 0]
            if grads:
                g = torch.cat(grads)
                gsnrs.append((g.mean().abs() / (g.std() + 1e-8)).item())
            self.optimizer.step()
            # 式(9) 原型动量更新
            self.proto_mem.update(emb.detach(), y)

            tot += total.item()
            for k in comps:
                comps[k] += comp[k]
            n_iter += 1
        n_iter = max(n_iter, 1)
        return {"loss": tot / n_iter, "ce": comps["ce"] / n_iter,
                "proto": comps["proto"] / n_iter, "reg": comps["reg"] / n_iter,
                "gsnr": float(np.mean(gsnrs)) if gsnrs else 0.0, "lr": lr}

    @torch.no_grad()
    def evaluate_acc(self, loader):
        """返回验证/测试 Top-1 准确率（用度量头 logits）。"""
        self.model.eval()
        correct = total = 0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            protos = self.proto_mem.get()
            logits_metric, emb, _ = self.model(x, prototypes=protos,
                                               tau_metric=self.tau_metric)
            if logits_metric is None:
                continue
            correct += (logits_metric.argmax(1) == y).sum().item()
            total += y.numel()
        return correct / max(total, 1)

    def run(self, train_loader, val_loader, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        history, best_acc, best_state, bad_epochs = [], 0.0, None, 0
        proto_history = []
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            m = self.train_one_epoch(train_loader, epoch)
            proto_history.append(self.proto_mem.get().cpu().numpy().copy())
            acc = self.evaluate_acc(val_loader)
            rec = {"epoch": epoch, "train_loss": m["loss"], "ce": m["ce"],
                   "proto": m["proto"], "reg": m["reg"], "gsnr": m["gsnr"],
                   "lr": m["lr"], "val_top1": acc, "time_s": round(time.time() - t0, 1)}
            history.append(rec)
            if epoch % self.cfg["train"]["log_every"] == 0 or epoch == 1:
                print(f"[Epoch {epoch:3d}] loss={m['loss']:.4f} ce={m['ce']:.3f} "
                      f"proto={m['proto']:.3f} gsnr={m['gsnr']:.2f} "
                      f"val_top1={acc*100:.1f}% lr={m['lr']:.2e} ({rec['time_s']}s)", flush=True)
            if acc > best_acc:
                best_acc, best_state, bad_epochs = acc, copy.deepcopy(self.model.state_dict()), 0
            else:
                bad_epochs += 1
                if bad_epochs >= 15:
                    print(f"Early stop at epoch {epoch} (val_top1={acc*100:.1f}%)")
                    break
        torch.save(best_state if best_state is not None else self.model.state_dict(),
                   os.path.join(out_dir, "best.pt"))
        with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        np.save(os.path.join(out_dir, "prototypes.npy"),
                self.proto_mem.get().cpu().numpy())
        np.save(os.path.join(out_dir, "proto_history.npy"), np.stack(proto_history))
        print(f"Training done. Best val Top-1 = {best_acc*100:.2f}% -> {out_dir}/best.pt")
        return history, best_state
