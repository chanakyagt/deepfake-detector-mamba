import os
CK = os.environ.get("DFM_CKPT_ROOT", "/tmp")
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from compare.wsp.harness import Session

p = torch.load(os.path.join(SC, "wsp_t1_pca.pt"))
mu, V, ev, k = p["mu"], p["V"], p["ev"], p["k"]
Vk, Vo = V[:, :k], V[:, k:]
s1 = torch.load(os.path.join(CK, "img_runs", "single_H1/model.pt"), map_location="cpu", weights_only=False)["model"]["proj.weight"]
W0 = torch.load(os.path.join(SC, "wsp_init_seed0.pt"))["proj.weight"]
S = Session("mamba_diloco")
W, b = S.state("proj.weight"), S.state("proj.bias")
for nm, M in (("diloco", W), ("single_H1", s1), ("init", W0)):
    print(nm, "largest gain inside 99% subspace", round(float(torch.linalg.svdvals(M @ Vk)[0]), 3),
          "largest gain in the 1% low-variance subspace", round(float(torch.linalg.svdvals(M @ Vo)[0]), 3),
          "Frobenius share outside", round(float((M @ Vo).norm() ** 2 / M.norm() ** 2), 3))
Wl = W @ Vo @ Vo.T
newW = W - Wl
newb = b + Wl @ mu
hyp = ("FRAGILE GAIN: proj.weight has gain up to 11.96 along input directions that carry only 1% of the training-feature "
       "variance (outside the top-821 principal components of 1,536 unlabelled practice features), versus 1.73 along the "
       "top PC; the init had a flat gain of ~0.36 everywhere, so this gain was grown by Adam/DiLoCo updates in directions "
       "with almost no training signal. A new domain that moves features along those directions gets that motion "
       "amplified 12x into the residual stream as noise. Edit: remove W's component on the low-variance subspace and fold "
       "its effect at the training-feature mean into proj.bias, so train-domain outputs move only by W(I-P)(x-mu).")
det = {"k_pcs": k, "practice_samples": p["n"], "variance_kept": 0.99, "AdaBN_like_disclosure":
       "the subspace and mean are statistics of unlabelled practice inputs (labels not used)"}
S.edit("proj.weight", newW, hyp, det)
S.edit("proj.bias", newb, "companion to the proj.weight edit: bias += W_removed @ mean(practice features)", det)
prev = torch.load(os.path.join(SC, "wsp_t1_prac.pt"))[0]
r = S.practice("E1 check on the same batch as pass 2", reuse=2)
d0 = prev["logits"][:, 1] - prev["logits"][:, 0]
d1 = r["logits"][:, 1] - r["logits"][:, 0]
print("pass-2 batch: acc before", float(((d0 > 0).long() == r["labels"]).float().mean()), "after", float(((d1 > 0).long() == r["labels"]).float().mean()),
      "mean |logit-diff change|", float((d1 - d0).abs().mean()), "corr", float(torch.corrcoef(torch.stack([d0, d1]))[0, 1]))
v = S.check_val("after E1 (low-variance subspace removed from proj)")
print("VAL", {k_: v[k_] for k_ in ("accuracy", "auc", "tp", "tn", "fp", "fn", "mean_p_fake_on_real", "mean_p_fake_on_fake")})
print("totals", S.totals(), "budget", S.budget.data["used"], S.budget.data["val_used"])
