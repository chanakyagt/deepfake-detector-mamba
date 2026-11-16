import os
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from compare.wsp.harness import Session

S = Session("mamba_diloco")
crit = ("pre-registered keep rule: keep only if val AUC >= baseline 0.9246 - 0.005; E1 val AUC 0.9166, acc 0.8365 -> REVERT. "
        "Correction to the E1 hypothesis text: the fair comparison is max gain inside the 99% subspace (17.64) vs outside (11.96), "
        "ratio 0.68 (single_H1 0.62, init 0.86), so the low-variance gain is not anomalous; and the val drop shows those "
        "directions carry in-domain signal.")
S.revert("proj.weight", crit)
S.revert("proj.bias", crit)
prev = torch.load(os.path.join(SC, "wsp_t1_prac.pt"))[0]
d0 = prev["logits"][:, 1] - prev["logits"][:, 0]
y = prev["labels"]


def report(r, tag):
    d1 = r["logits"][:, 1] - r["logits"][:, 0]
    print(tag, "acc before", float(((d0 > 0).long() == y).float().mean()), "after", float(((d1 > 0).long() == y).float().mean()),
          "mean|ld| before", float(d0.abs().mean()), "after", float(d1.abs().mean()),
          "corr", float(torch.corrcoef(torch.stack([d0, d1]))[0, 1]),
          "rank-AUC on batch", float(((d1[y == 1][:, None] > d1[y == 0][None]).float().mean())), "vs",
          float(((d0[y == 1][:, None] > d0[y == 0][None]).float().mean())))


init = torch.load(os.path.join(SC, "wsp_init_seed0.pt"))
alpha = 1 / 2.7157
names = [n for n, _, _, isp in S.names() if isp]
for n in names:
    w = S.state(n)
    S.edit(n, init[n] + alpha * (w - init[n]),
           "DILOCO OVERSHOOT: the decay-only A_log tensors shrank 5.13x more than in single_H1 (0.9543 vs 0.9910 of init), a direct "
           "readout of the outer Nesterov step amplifying every displacement; total |W-W0| is 2.716x single_H1's. Edit: scale every "
           f"trained displacement by alpha=1/2.7157={alpha:.4f} (matching single_H1's displacement norm), W = W0 + alpha (W - W0), "
           "W0 rebuilt exactly from seed 0.", {"alpha": alpha})
report(S.practice("E2a (shrink displacement to single_H1 norm) on batch of pass 2", reuse=2), "E2a")
for n in names:
    S.revert(n, "E2a practice probe finished")
c = S.state("layers.1.mixer.x_proj.weight")
c[16:] = 0
S.edit("layers.1.mixer.x_proj.weight", c,
       "CLASS-AGNOSTIC HIGH-GAIN PATH: in layer 1 the selective path dt*(B.C)*u has 11x the norm of the skip path D*u "
       "(98.75 vs 8.98) and makes the layer-1 mixer the largest write to the residual stream (99 of 156), yet its per-sample "
       "gain B.C is uncorrelated with the logit (r=+0.002) and heavy-tailed (to -75.7). Edit: zero the 16 C rows of "
       "layers.1.mixer.x_proj so B.C=0 and only D*u remains.")
report(S.practice("E2b (layer-1 selective path off) on batch of pass 2", reuse=2), "E2b")
S.revert("layers.1.mixer.x_proj.weight", "E2b practice probe finished")
print("totals", S.totals(), "budget", S.budget.data["used"], S.budget.data["val_used"])
