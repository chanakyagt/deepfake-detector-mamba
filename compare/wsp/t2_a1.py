import os
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import torch
outs = torch.load(os.path.join(SC, "wsp_t2_survey.pt"))
c = torch.load(os.path.expanduser("~/sota/dfb_weights/effnb4_best.pth"), map_location="cpu", weights_only=True)
c = c.get("state_dict", c)
c = {k[7:] if k.startswith("module.") else k: v for k, v in c.items()}
y = torch.cat([o["labels"] for o in outs])
L = torch.cat([o["logits"] for o in outs])
pen = torch.cat([o["pen"] for o in outs])
ld = L[:, 1] - L[:, 0]
print("N", len(y), "logit diff real", float(ld[y == 0].mean()), float(ld[y == 0].std()), "fake", float(ld[y == 1].mean()), float(ld[y == 1].std()))
print("batch AUC on practice", float((ld[y == 1][:, None] > ld[y == 0][None]).float().mean()))
names = list(outs[0]["bn"].keys())
rows = []
for n in names:
    s = sum(o["bn"][n][0] for o in outs); q = sum(o["bn"][n][1] for o in outs); cnt = sum(o["bn"][n][2] for o in outs)
    m = s / cnt; v = q / cnt - m * m
    rm, rv = c["backbone." + n.split("backbone.", 1)[1] + ".running_mean"], c["backbone." + n.split("backbone.", 1)[1] + ".running_var"]
    z = ((m - rm) / (rv + 1e-3).sqrt()).abs().mean()
    vr = (v / (rv + 1e-3)).log().abs().mean().exp()
    rows.append((n, float(z), float(vr), float((v / (rv + 1e-3)).median())))
for r in rows[:6] + rows[len(rows)//2-2:len(rows)//2+2] + rows[-6:]:
    print(f"{r[0]:55s} mean|z-shift|={r[1]:.3f} geo-mean var-ratio factor={r[2]:.3f} median var ratio={r[3]:.3f}")
print("layers with mean|z|>0.5:", sum(r[1] > 0.5 for r in rows), "of", len(rows), " worst", max(rows, key=lambda r: r[1])[:2])
W = c["backbone.last_layer.weight"]; b = c["backbone.last_layer.bias"]
w = W[1] - W[0]
contrib = pen * w
sep = contrib[y == 1].mean(0) - contrib[y == 0].mean(0)
tot = sep.abs().sum()
o = sep.abs().argsort(descending=True)
print("head bias diff", float(b[1] - b[0]), "|w|", float(w.norm()), "mean logit-diff", float(ld.mean()))
print("practice separation carried by top 5/20/100 of 1792 penultimate channels:", [round(100 * float(sep.abs()[o[:k]].sum() / tot), 1) for k in (5, 20, 100)])
print("net separation (signed sum) vs abs sum", float(sep.sum()), float(tot))
mean_c = contrib.mean(0)
print("channels with largest mean |contribution| to logit diff (any class):", [(int(j), round(float(mean_c[j]), 3), round(float(sep[j]), 3)) for j in mean_c.abs().argsort(descending=True)[:8]])
