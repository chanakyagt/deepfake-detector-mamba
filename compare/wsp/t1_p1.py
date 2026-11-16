import os
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from compare.wsp.harness import Session

S = Session("mamba_diloco")
v = S.check_val("baseline: original diloco_H100 weights, no edits")
print("VAL", {k: v[k] for k in ("accuracy", "auc", "tp", "tn", "fp", "fn", "mean_p_fake_on_real", "mean_p_fake_on_fake")})
hooks = ["proj"] + [f"layers.{i}" for i in range(4)] + [f"layers.{i}.mixer" for i in range(4)] + ["norm_f", "head.0", "head.1"]
outs = []
for k in range(8):
    r = S.practice("activation survey: residual stream, mixer outputs, head hidden, logits", hooks=hooks)
    outs.append(r)
    acc = float((r["logits"].argmax(1) == r["labels"]).float().mean())
    print("practice", k, "acc", acc, "mean|logit diff|", float((r["logits"][:, 1] - r["logits"][:, 0]).abs().mean()))
torch.save(outs, os.path.join(SC, "wsp_t1_prac.pt"))
print("budget", S.budget.data["used"], "val", S.budget.data["val_used"])
