import os
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import torch
import torch.nn as nn
from compare.wsp.harness import Session

S = Session("dfb_effnetb4")
print("load", S.t.load_res, "params", S.total_params, "buffers", S.total_buffers)
bns = [n for n, m in S.model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
print("BN layers", len(bns), "momentum/eps", S.module(bns[0]).momentum, S.module(bns[0]).eps)
v = S.check_val("baseline: original FF++ weights, zero-shot")
print("VAL", {k: v[k] for k in ("accuracy", "auc", "tp", "tn", "fp", "fn", "mean_p_fake_on_real", "mean_p_fake_on_fake")})
outs = []
for k in range(4):
    r = S.practice("survey (eval mode): logits, penultimate features, BN input moments vs stored running stats",
                   inputs_of=["backbone.last_layer"], bn_moments=True)
    r["bn_moments"] = {n: (a, b, c) for n, (a, b, c) in r["bn_moments"].items()}
    outs.append({"logits": r["logits"], "labels": r["labels"], "names": r["names"], "pen": r["acts"]["in:backbone.last_layer"],
                 "bn": r["bn_moments"]})
    acc = float(((r["p_fake"] > 0.5).long() == r["labels"]).float().mean())
    print("practice", k, "acc", acc, "mean p_fake real/fake", float(r["p_fake"][r["labels"] == 0].mean()), float(r["p_fake"][r["labels"] == 1].mean()))
torch.save(outs, os.path.join(SC, "wsp_t2_survey.pt"))
print("budget", S.budget.data["used"], S.budget.data["val_used"])
print("peak GPU GB", torch.cuda.max_memory_allocated() / 1e9)
