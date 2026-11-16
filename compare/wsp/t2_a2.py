import os
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import torch
outs = torch.load(os.path.join(SC, "wsp_t2_survey.pt"))
c = torch.load(os.path.expanduser("~/sota/dfb_weights/effnb4_best.pth"), map_location="cpu", weights_only=True)
c = c.get("state_dict", c)
c = {k[7:] if k.startswith("module.") else k: v for k, v in c.items()}
y = torch.cat([o["labels"] for o in outs])
pen = torch.cat([o["pen"] for o in outs])
print("pen shape", pen.shape, "min", float(pen.min()), "mean", float(pen.mean()), "frac exactly 0", float((pen == 0).float().mean()))
W = c["backbone.last_layer.weight"]; w = W[1] - W[0]
gap = pen[y == 1].mean(0) - pen[y == 0].mean(0)
sep = w * gap
print("frac sep>0", float((sep > 0).float().mean()), "sum", float(sep.sum()), "abs sum", float(sep.abs().sum()))
for n in ["backbone.efficientnet._blocks.31._bn1", "backbone.efficientnet._blocks.16._bn1", "backbone.efficientnet._bn1", "backbone.efficientnet._blocks.4._bn2"]:
    s = sum(o["bn"][n][0] for o in outs); q = sum(o["bn"][n][1] for o in outs); cnt = sum(o["bn"][n][2] for o in outs)
    m = s / cnt; v = q / cnt - m * m
    rm, rv = c[n + ".running_mean"], c[n + ".running_var"]
    print(n, "cnt", cnt, "obs mean range", float(m.min()), float(m.max()), "obs var median", float(v.median()), "min", float(v.min()),
          "| running mean range", float(rm.min()), float(rm.max()), "running var median", float(rv.median()), "max", float(rv.max()))
ks = [k for k in c if k.endswith("running_var")]
big = sorted(((float(c[k].median()), k) for k in ks), reverse=True)[:5]
print("BN layers with largest median running_var:", big)
