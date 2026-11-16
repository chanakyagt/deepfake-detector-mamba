import os
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
import torch.nn.functional as F
from compare.wsp.harness import Session

S = Session("mamba_diloco")
hooks = []
for i in range(4):
    hooks += [f"layers.{i}.mixer.{m}" for m in ("in_proj", "conv1d", "x_proj", "dt_proj")]
r = S.practice("mixer internals: split each 1-frame mixer output into the selective path dt*(B.C)*u and the skip path D*u", hooks=hooks,
               reuse=2)
torch.save(r, os.path.join(SC, "wsp_t1_mix.pt"))
y = r["labels"]
for i in range(4):
    a = r["acts"]
    up, z = a[f"layers.{i}.mixer.in_proj"][:, 0].chunk(2, -1)
    u = F.silu(a[f"layers.{i}.mixer.conv1d"][:, :, 0])
    dt = F.softplus(a[f"layers.{i}.mixer.dt_proj"][:, 0])
    Bm, Cm = a[f"layers.{i}.mixer.x_proj"][:, 0].chunk(2, -1)
    bc = (Bm * Cm).sum(-1, keepdim=True)
    D = S.state(f"layers.{i}.mixer.D")
    sel = dt * bc * u
    skp = D * u
    g = F.silu(z)
    print(f"layer {i}: |sel path| {float((sel*g).norm(dim=1).mean()):.3f}  |skip path| {float((skp*g).norm(dim=1).mean()):.3f}  "
          f"dt median {float(dt.median()):.4f} max {float(dt.max()):.3f}  B.C mean {float(bc.mean()):.3f} std {float(bc.std()):.3f} "
          f"range [{float(bc.min()):.2f},{float(bc.max()):.2f}]  B.C real/fake means {float(bc[y==0].mean()):.3f}/{float(bc[y==1].mean()):.3f}")
