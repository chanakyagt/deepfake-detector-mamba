import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from compare.wsp.harness import Session

S = Session("dfb_effnetb4")
lds = []
for k in range(3):
    r = S.practice("unlabelled logit survey after AdaBN (labels not used): where does the fake-minus-real logit sit?", balanced=False)
    lds.append(r["logits"][:, 1] - r["logits"][:, 0])
ld = torch.cat(lds)
med = float(ld.median())
print("practice logit-diff after AdaBN: median", med, "mean", float(ld.mean()), "frac>0", float((ld > 0).float().mean()))
b = S.state("backbone.last_layer.bias")
b[1] = b[1] - med
S.edit("backbone.last_layer.bias", b,
       "MISCALIBRATED BIAS AFTER AdaBN: with adapted BN statistics the val p_fake moved to ~0.77 for both classes (val acc 0.513 at "
       "AUC 0.549), i.e. the decision threshold no longer sits between the classes. The Train split is 50% fake by construction, "
       f"so the label-free median of the fake-minus-real logit over 192 unlabelled practice images ({med:.4f}) should be the "
       "threshold. Edit: subtract that median from the fake bias (one weight). Ranking (AUC) is unaffected by construction.",
       {"median_logit_diff": med, "n_images": int(ld.numel())})
v = S.check_val("after AdaBN + label-free bias recentring")
print("VAL", {k_: v[k_] for k_ in ("accuracy", "auc", "tp", "tn", "fp", "fn", "mean_p_fake_on_real", "mean_p_fake_on_fake")})
print("totals", S.totals(), "budget", S.budget.data["used"], S.budget.data["val_used"])
