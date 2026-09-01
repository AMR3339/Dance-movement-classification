"""冒烟测试：数据生成 / 预处理 / 模型前向 / 损失 / 原型更新。"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import yaml

from data.synth_data import generate_dataset
from data.preprocessing import preprocess_pipeline, build_beat_template, motion_energy
from data.dataset import make_loaders
from models.ms_g3d import MSG3D
from models.losses import PrototypeMemory, prototype_contrast_loss, class_weights

BASE = os.path.dirname(os.path.abspath(__file__))
cfg = yaml.safe_load(open(os.path.join(BASE, "config.yaml"), encoding="utf-8"))

# 数据
ds = generate_dataset(cfg, cache_dir=os.path.join(BASE, "data_cache"))
data, labels, dancer_ids = ds["data"], ds["labels"], ds["dancer_ids"]
print("data:", data.shape, "labels:", labels.shape, "classes:", len(set(labels.tolist())))
tr_loader, te_loader, tr_idx, te_idx = make_loaders(data, labels, dancer_ids, cfg, batch_size=16)
print("train:", len(tr_idx), "test:", len(te_idx))

# 预处理（小批量）
data_tvc = np.transpose(data, (0, 3, 1, 2))
energy = [motion_energy(data_tvc[i]) for i in tr_idx[:20]]
tmpl, period = build_beat_template(energy)
print("beat template period:", period)
t0 = time.time()
proc, shifts = preprocess_pipeline(data[:16], align=True, template=tmpl, period=period)
print(f"preprocess {proc.shape} {time.time()-t0:.2f}s, shifts range: [{shifts.min()}, {shifts.max()}]")

# 模型
model = MSG3D(cfg)
print("params: %.2f M" % (sum(p.numel() for p in model.parameters()) / 1e6))
x = torch.from_numpy(proc[:4]).float()
t0 = time.time()
protos = torch.randn(cfg["model"]["num_classes"], cfg["model"]["emb_dim"])
protos = protos / protos.norm(dim=1, keepdim=True)
logits, emb, mats = model(x, prototypes=protos, tau_metric=0.2)
print(f"forward: logits {tuple(logits.shape)} emb {tuple(emb.shape)} mats {len(mats)} ({time.time()-t0:.2f}s)")

# 损失 + 原型
y = torch.from_numpy(labels[:4])
l_proto = prototype_contrast_loss(emb, y, protos, tau=0.2, k=3)
print("proto loss:", l_proto.item())
pm = PrototypeMemory(cfg["model"]["num_classes"], cfg["model"]["emb_dim"])
pm.update(emb.detach(), y)
print("prototype updated, norm:", pm.get().norm(dim=1).mean().item())
w = class_weights(torch.from_numpy(labels), cfg["model"]["num_classes"])
print("ce weights sum:", w.sum().item())
print("\nSMOKE TEST OK")
