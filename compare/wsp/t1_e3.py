import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from compare.wsp.harness import Session

S = Session("mamba_diloco")
c = S.state("layers.1.mixer.x_proj.weight")
c[16:] = 0
S.edit("layers.1.mixer.x_proj.weight", c,
       "KEEP-CANDIDATE E2b: the layer-1 selective path dt*(B.C)*u is the largest write into the residual stream (mixer out 99 vs "
       "stream 156 after the layer) but is class-agnostic (B.C vs logit r=+0.002); switching it off on the pass-2 practice batch "
       "left accuracy unchanged (0.8906 -> 0.8906), logit-diff correlation 0.994, batch AUC 0.9844 -> 0.9834. Because norm_f "
       "divides by the whole stream norm, a large, heavy-tailed, label-free component sets the per-image scale of the "
       "discriminative signal; if its magnitude moves under a domain shift it rescales evidence arbitrarily. Removing it "
       "should cost nothing in-domain and remove one source of shift sensitivity. Edit: zero the C rows (16 x 1024).")
v = S.check_val("after E2b (layer-1 selective path off)")
print("VAL", {k_: v[k_] for k_ in ("accuracy", "auc", "tp", "tn", "fp", "fn", "mean_p_fake_on_real", "mean_p_fake_on_fake")})
print("totals", S.totals(), "budget", S.budget.data["used"], S.budget.data["val_used"])
