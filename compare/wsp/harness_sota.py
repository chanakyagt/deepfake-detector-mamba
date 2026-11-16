import os
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness as H
from harness import Session, BudgetExceeded, summary, now, weight_stats

HOME = Path.home()
REPO = H.REPO
UNUSED_IN_FORWARD = ("U_r", "S_r", "V_r")


class _ImageTarget:
    size = None
    amp = False

    def _setup(self, model, cfg, res, device, batch, workers):
        sys.path.insert(0, str(REPO / "compare"))
        from sota_zero_shot import Images
        from dfmamba.extract_images import ImageFolderFaces
        self.Images, self.Folder = Images, ImageFolderFaces
        self.cfg = cfg
        self.load_res = {"missing": len(res.missing_keys), "unexpected": len(res.unexpected_keys),
                         "missing_sample": list(res.missing_keys[:5]), "unexpected_sample": list(res.unexpected_keys[:5])}
        self.dev = torch.device(device)
        self.model = model.to(self.dev).eval()
        self.images = str(HOME / "data/MegaSub")
        self.size = cfg["resolution"]
        self.mean = torch.tensor(cfg["mean"], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(cfg["std"], device=self.dev).view(1, 3, 1, 1)
        self.batch, self.workers = batch, min(workers, 4)
        self._train = None

    def build_copy(self):
        raise RuntimeError("deep copies are not used by harness_sota; final swaps changed tensors instead")

    def train_items(self):
        if self._train is None:
            self._train = self.Folder(self.images, "Train").items
        return self._train

    def n_train(self):
        return len(self.train_items())

    def train_labels(self):
        return torch.tensor([l for _, l in self.train_items()])

    def norm(self, x):
        return (x.to(self.dev).float() / 255.0 - self.mean) / self.std

    def practice_inputs(self, idx):
        items = [self.train_items()[i] for i in idx]
        ds = self.Images(items, self.size, False)
        xs = torch.stack([ds[i][0] for i in range(len(ds))])
        return self.norm(xs), torch.tensor([l for _, l in items]), [str(p) for p, _ in items]

    def ctx(self):
        if self.amp:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def logits(self, model, x):
        with self.ctx():
            return self._logits(model, x).float()

    def loader(self, split):
        from torch.utils.data import DataLoader
        if split == "Test" and not getattr(self, "_final_ok", False):
            raise BudgetExceeded("Test images may only be loaded by final()")
        items = self.Folder(self.images, split).items
        return items, DataLoader(self.Images(items, self.size, False), batch_size=self.batch,
                                 num_workers=self.workers, pin_memory=False)

    def score_split(self, models, split):
        items, dl = self.loader(split)
        ys, out = [], [[] for _ in models]
        for x, y in dl:
            xn = self.norm(x)
            for k, m in enumerate(models):
                out[k].append(self.logits(m, xn).softmax(1)[:, 1].cpu())
            ys.append(y)
        return [torch.cat(o).numpy() for o in out], torch.cat(ys).numpy(), len(items)


class EffortTarget(_ImageTarget):
    name = "effort"
    amp = True

    def __init__(self, device="cuda", batch=32, workers=4):
        sys.path.insert(0, str(REPO / "compare"))
        from sota_zero_shot import build_effort
        model, cfg, res = build_effort(str(HOME / "sota/Effort/DeepfakeBench/training"),
                                       str(HOME / "sota/effort_weights/effort_clip_L14_trainOn_FaceForensic.pth"),
                                       str(HOME / "sota/hf_clip/clip-vit-large-patch14"), str(HOME / "sota/pydeps"))
        self._setup(model, cfg, res, device, batch, workers)

    def _logits(self, model, x):
        return model({"image": x}, inference=True)["cls"]


class DFBNamed(_ImageTarget):
    dfb_name = None

    def __init__(self, device="cuda", batch=32, workers=4):
        sys.path.insert(0, str(REPO / "compare"))
        from sota_zero_shot import build_dfb
        model, cfg, res = build_dfb(self.dfb_name, str(HOME / "sota/DeepfakeBench/training"),
                                    str(HOME / "sota/dfb_weights"), os.environ.get("DFM_SCRATCH", "/tmp"), str(HOME / "sota/pydeps"))
        self._setup(model, cfg, res, device, batch, workers)

    def _logits(self, model, x):
        out = model({"image": x, "label": torch.zeros(x.shape[0], dtype=torch.long, device=x.device)},
                    inference=True)["cls"]
        if hasattr(model, "prob"):
            model.prob, model.label = [], []
        return out


class UCFTarget(DFBNamed):
    name = "ucf"
    dfb_name = "ucf"


class XceptionTarget(DFBNamed):
    name = "xception"
    dfb_name = "xception"


SOTA = {"effort": EffortTarget, "ucf": UCFTarget, "xception": XceptionTarget}
H.TARGETS.update(SOTA)


class SotaSession(Session):
    def __init__(self, target, **kw):
        if target not in SOTA:
            raise ValueError(target)
        super().__init__(target, **kw)
        self.forward_params = sum(p.numel() for n, p in self.model.named_parameters()
                                  if n.rsplit(".", 1)[-1] not in UNUSED_IN_FORWARD)

    def totals(self, over=None):
        t = super().totals(over)
        fp = getattr(self, "forward_params", None)
        if fp:
            t["forward_used_params"] = fp
            t["pct_forward_used_params"] = 100.0 * t["changed_params"] / fp
        return t

    def changed_keys(self):
        sd = self.model.state_dict()
        return [k for k in sd if not torch.equal(sd[k].detach().cpu(), self.base[k])]

    def final(self):
        fpath = self.dir / "final.json"
        if fpath.exists() or self.budget.data.get("final"):
            raise BudgetExceeded("final already run for this target")
        self.budget.data["final"] = {"time": now(), "status": "started"}
        self.budget.save()
        sd = self.model.state_dict()
        keys = self.changed_keys()
        edited = {k: sd[k].detach().clone() for k in keys}
        orig = {k: self.base[k].to(sd[k].device, sd[k].dtype) for k in keys}

        def put(vals):
            for k, v in vals.items():
                sd[k].copy_(v)

        self.t._final_ok = True
        items, dl = self.t.loader("Test")
        ys, pb, pe = [], [], []
        try:
            for x, y in dl:
                xn = self.t.norm(x)
                put(orig)
                pb.append(self.t.logits(self.model, xn).softmax(1)[:, 1].cpu())
                put(edited)
                pe.append(self.t.logits(self.model, xn).softmax(1)[:, 1].cpu())
                ys.append(y)
        finally:
            put(edited)
            self.t._final_ok = False
        y = torch.cat(ys).numpy()
        pb, pe = torch.cat(pb).numpy(), torch.cat(pe).numpy()
        out = {"target": self.target_name, "split": "Test", "n": len(items), "time": now(),
               "original_weights": summary(y, pb), "edited_weights": summary(y, pe),
               "forward_passes_used": self.budget.data["used"], "val_checks_used": self.budget.data["val_used"],
               "edits": len(self.edit_log()), "changed_tensors": keys, "totals": self.totals(),
               "note": "single read of Test; each batch scored with original then edited weights (changed tensors swapped in place)"}
        for k, p in (("original_weights", pb), ("edited_weights", pe)):
            out[k]["mean_p_fake_on_real"] = float(p[y == 0].mean())
            out[k]["mean_p_fake_on_fake"] = float(p[y == 1].mean())
        fpath.write_text(json.dumps(out, indent=1))
        self.budget.data["final"] = {"time": now(), "status": "done"}
        self.budget.save()
        return out


def save_pass(out, path):
    np.savez(path, logits=out["logits"].numpy(), labels=out["labels"].numpy(), indices=np.asarray(out["indices"]),
             **{"act__" + k.replace(".", "__").replace(":", "_"): v.numpy() for k, v in out["acts"].items()})
