import os
CK = os.environ.get("DFM_CKPT_ROOT", "/tmp")
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from dfmamba.model import temporal_from_cfg

c = torch.load(os.path.join(CK, "img_runs", "diloco_H100/model.pt"), map_location="cpu", weights_only=False)
s = torch.load(os.path.join(CK, "img_runs", "single_H1/model.pt"), map_location="cpu", weights_only=False)
print(s["config"]["train"])
torch.manual_seed(c["config"]["train"]["seed"])
cfg = dict(c["config"])
cfg["model"] = dict(cfg["model"])
cfg["model"].setdefault("feat_dim", 1280)
m0 = temporal_from_cfg(cfg).state_dict()
torch.save(m0, os.path.join(SC, "wsp_init_seed0.pt"))
tot_d = tot_s = 0.0
for k in m0:
    w0, wd, ws = m0[k].float(), c["model"][k].float(), s["model"][k].float()
    dd, ds = (wd - w0).flatten(), (ws - w0).flatten()
    cos = float(torch.nn.functional.cosine_similarity(dd, ds, 0)) if dd.norm() > 0 and ds.norm() > 0 else float("nan")
    tot_d += float(dd.norm() ** 2); tot_s += float(ds.norm() ** 2)
    print(f"{k:32s} |W0|={w0.norm():8.3f} |dW_diloco|={dd.norm():8.3f} |dW_single|={ds.norm():8.3f} ratio={float(dd.norm()/(ds.norm()+1e-12)):6.2f} cos(dW_d,dW_s)={cos:.3f}")
print("total delta norm diloco", tot_d ** 0.5, "single", tot_s ** 0.5, "ratio", (tot_d / tot_s) ** 0.5)
