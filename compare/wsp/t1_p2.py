import os
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from compare.wsp.harness import Session

S = Session("mamba_diloco")
prev = torch.load(os.path.join(SC, "wsp_t1_prac.pt"))[0]
init = torch.load(os.path.join(SC, "wsp_init_seed0.pt"))
for l in range(4):
    S.edit(f"layers.{l}.mixer.A_log", init[f"layers.{l}.mixer.A_log"],
           "CONTROL: with 1-frame inputs the scan starts from h=0, so exp(dt*A)*h is always 0 and A_log cannot affect the output; "
           "its trained value is exactly 0.9543 x its init (weight decay only, no gradient). Restoring init must leave logits bit-identical.")
r = S.practice("control: same batch as pass 2 after restoring A_log to init in all 4 layers", reuse=2)
d = float((r["logits"] - prev["logits"]).abs().max())
print("A_log control: max |logit change| =", d)
for l in range(4):
    S.revert(f"layers.{l}.mixer.A_log", f"control finished (max |logit change| on the reused batch = {d})")
xs = []
for k in range(16):
    r = S.practice("feature-covariance survey: unlabelled statistics of the 1280-d EfficientNet input features", balanced=False)
    xs.append((r["inputs"][:, 0], r["labels"], r["logits"]))
torch.save(xs, os.path.join(SC, "wsp_t1_cov.pt"))
print("budget", S.budget.data["used"], "totals", S.totals())
