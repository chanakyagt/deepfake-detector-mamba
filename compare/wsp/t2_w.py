import os
import torch
c = torch.load(os.path.expanduser("~/sota/dfb_weights/effnb4_best.pth"), map_location="cpu", weights_only=True)
c = c.get("state_dict", c)
c = {k[7:] if k.startswith("module.") else k: v for k, v in c.items()}
tot = dead = 0
per = []
for k in c:
    if k.endswith("running_var"):
        base = k[:-len("running_var")]
        rv, g = c[k], c[base + "weight"]
        d = int(((rv < 1e-6) & (g.abs() < 1e-3)).sum())
        tot += rv.numel(); dead += d
        per.append((d / rv.numel(), d, rv.numel(), base))
print("BN channels with running_var<1e-6 and |gamma|<1e-3:", dead, "of", tot)
for r in sorted(per, reverse=True)[:8]:
    print(f"{r[3]:50s} dead {r[1]}/{r[2]} ({100*r[0]:.1f}%)")
W = c["backbone.last_layer.weight"]; b = c["backbone.last_layer.bias"]
print("head W rows norms", [float(x) for x in W.norm(dim=1)], "cos(rows)", float(torch.nn.functional.cosine_similarity(W[0], W[1], 0)), "bias", b.tolist())
fk = c["backbone.efficientnet._bn1.weight"]
print("final BN gamma: frac |gamma|<1e-3", float((fk.abs() < 1e-3).float().mean()))
w = W[1] - W[0]
print("head diff weights on final-BN-dead channels |w| sum", float(w[(fk.abs() < 1e-3)].abs().sum()), "total", float(w.abs().sum()))
