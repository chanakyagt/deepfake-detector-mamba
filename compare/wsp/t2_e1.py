import os
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
from compare.wsp.harness import Session

S = Session("dfb_effnetb4")
NP = int(sys.argv[1]) if len(sys.argv) > 1 else 16
acc = {}
for k in range(NP):
    r = S.practice("AdaBN statistics: BN layers normalise with batch statistics (running stats untouched), record BN input moments; labels unused",
                   balanced=False, bn_batch_stats=True, bn_moments=True)
    for n, (s, q, c) in r["bn_moments"].items():
        a = acc.setdefault(n, [0, 0, 0])
        a[0] = a[0] + s; a[1] = a[1] + q; a[2] += c
torch.save(acc, os.path.join(SC, "wsp_t2_adabn_moments.pt"))
hyp = ("BN MISMATCH: the 96 BatchNorm layers normalise with running mean/var estimated on FF++ c23 video frames; on this "
       "dataset's Train practice images the per-channel input means sit far from those statistics (survey passes 2-5, see log). "
       "Every downstream layer then sees inputs shifted/rescaled relative to what it was trained on. Edit (AdaBN-like, disclosed): "
       f"replace running_mean/running_var with the pooled moments of BN inputs over {NP} x 64 unlabelled Train practice images, "
       "collected with batch-statistics normalisation so deeper layers see adapted inputs.")
for n, (s, q, c) in acc.items():
    m = s / c
    v = (q / c - m * m) * c / (c - 1)
    S.edit(n + ".running_mean", m, hyp, {"images": NP * 64, "count_per_channel": c})
    S.edit(n + ".running_var", v.clamp(min=0), hyp, {"images": NP * 64, "count_per_channel": c})
r = S.practice("AdaBN check on the batch of pass 2", reuse=2, inputs_of=["backbone.last_layer"])
torch.save({"logits": r["logits"], "labels": r["labels"], "pen": r["acts"]["in:backbone.last_layer"]}, os.path.join(SC, "wsp_t2_after_adabn.pt"))
y = r["labels"]; ld = r["logits"][:, 1] - r["logits"][:, 0]
print("pass-2 batch after AdaBN: acc", float(((ld > 0).long() == y).float().mean()), "batch AUC",
      float((ld[y == 1][:, None] > ld[y == 0][None]).float().mean()), "mean p_fake real/fake",
      float(r["p_fake"][y == 0].mean()), float(r["p_fake"][y == 1].mean()))
v = S.check_val("after AdaBN re-estimation of all BN running stats")
print("VAL", {k_: v[k_] for k_ in ("accuracy", "auc", "tp", "tn", "fp", "fn", "mean_p_fake_on_real", "mean_p_fake_on_fake")})
print("totals", S.totals(), "budget", S.budget.data["used"], S.budget.data["val_used"])
print("peak GPU GB", torch.cuda.max_memory_allocated() / 1e9)
