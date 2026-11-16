import os
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import torch
import torch.nn.functional as F
r = torch.load(os.path.join(SC, "wsp_t1_mix.pt"))
a, y = r["acts"], r["labels"]
ld = r["logits"][:, 1] - r["logits"][:, 0]
for i in range(4):
    dt = F.softplus(a[f"layers.{i}.mixer.dt_proj"][:, 0])
    Bm, Cm = a[f"layers.{i}.mixer.x_proj"][:, 0].chunk(2, -1)
    bc = (Bm * Cm).sum(-1)
    q = torch.quantile(bc, torch.tensor([.05, .25, .5, .75, .95]))
    corr = torch.corrcoef(torch.stack([bc, ld]))[0, 1]
    cm = dt.max(0).values
    print(f"layer {i}: B.C quantiles {[round(float(v),2) for v in q]}, corr(B.C, logit diff) {float(corr):+.3f}; "
          f"channels whose max dt > 1: {int((cm > 1).sum())}/1024, mean dt of those {float(dt[:, cm > 1].mean()):.3f}; "
          f"frac of (dt) mass in top-16 channels {float(dt.mean(0).topk(16).values.sum() / dt.mean(0).sum()):.3f}")
    sel_mag = (dt * bc[:, None]).abs().mean(1)
    print("   samples sorted by selective gain: top-8 labels", y[sel_mag.argsort(descending=True)[:8]].tolist(), "logit diffs",
          [round(float(v), 1) for v in ld[sel_mag.argsort(descending=True)[:8]]])
