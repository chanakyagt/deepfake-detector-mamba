import os
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from dfmamba.metrics import summary

RESULTS = REPO / "compare" / "results" / "wsp"
PASS_LIMIT = 50
VAL_LIMIT = 5
HOME = Path.home()


class BudgetExceeded(RuntimeError):
    pass


class Banned(RuntimeError):
    pass


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Budget:
    def __init__(self, d):
        self.path = d / "budget.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {"limit": PASS_LIMIT, "val_limit": VAL_LIMIT, "used": 0, "val_used": 0, "passes": [],
                         "final": None}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=1))

    def charge(self, kind, n, note, extra=None):
        if self.data["used"] >= PASS_LIMIT:
            raise BudgetExceeded(f"pass {self.data['used'] + 1} refused: budget of {PASS_LIMIT} forward passes used")
        if kind == "val" and self.data["val_used"] >= VAL_LIMIT:
            raise BudgetExceeded(f"validation check refused: {VAL_LIMIT} checks used")
        self.data["used"] += 1
        if kind == "val":
            self.data["val_used"] += 1
        rec = {"pass": self.data["used"], "kind": kind, "n": n, "note": note, "time": now()}
        if extra:
            rec.update(extra)
        self.data["passes"].append(rec)
        self.save()
        return rec


def _no_grad_guard():
    torch.set_grad_enabled(False)


class MambaTarget:
    name = "mamba_diloco"

    def __init__(self):
        from dfmamba.model import temporal_from_cfg
        ck = Path(os.environ.get("DFM_CKPT_ROOT", "/tmp")) / "img_runs" / "diloco_H100" / "model.pt"
        if not ck.exists():
            ck = Path(os.environ.get("DFM_CKPT_ROOT", "/tmp")) / "img_runs" / "single_H1" / "model.pt"
        self.ckpt = str(ck)
        c = torch.load(ck, map_location="cpu", weights_only=False)
        self.model = temporal_from_cfg(c["config"])
        self.model.load_state_dict(c["model"])
        self.model.eval()
        self.feats_dir = Path(os.environ.get("DFM_FEATS_ROOT", "/tmp")) / "mega_feats"
        self._train = None

    def build_copy(self):
        return copy.deepcopy(self.model).eval()

    def train_split(self):
        if self._train is None:
            self._train = torch.load(self.feats_dir / "train.pt", map_location="cpu")
        return self._train

    def n_train(self):
        return len(self.train_split()["labels"])

    def train_labels(self):
        return self.train_split()["labels"]

    def practice_inputs(self, idx):
        d = self.train_split()
        return d["feats"][idx].float(), d["labels"][idx], [d["names"][i] for i in idx]

    def logits(self, model, x):
        return model(x)

    def score_split(self, models, split):
        d = torch.load(self.feats_dir / f"{split}.pt", map_location="cpu")
        x, y = d["feats"], d["labels"]
        out = [[] for _ in models]
        for s in range(0, len(y), 2048):
            xb = x[s:s + 2048].float()
            for k, m in enumerate(models):
                out[k].append(m(xb).float().softmax(1)[:, 1])
        return [torch.cat(o).numpy() for o in out], y.numpy(), len(y)


class DFBTarget:
    name = "dfb_effnetb4"

    def __init__(self, device="cuda", batch=32, workers=4):
        sys.path.insert(0, str(REPO / "compare"))
        from sota_zero_shot import build_dfb, Images
        from dfmamba.extract_images import ImageFolderFaces
        self.Images, self.Folder = Images, ImageFolderFaces
        self.model, self.cfg, res = build_dfb("efficientnetb4", str(HOME / "sota/DeepfakeBench/training"),
                                              str(HOME / "sota/dfb_weights"), os.environ.get("DFM_SCRATCH", "/tmp"), str(HOME / "sota/pydeps"))
        self.load_res = {"missing": len(res.missing_keys), "unexpected": len(res.unexpected_keys)}
        self.dev = torch.device(device)
        self.model = self.model.to(self.dev).eval()
        self.images = str(HOME / "data/MegaSub")
        self.size = self.cfg["resolution"]
        self.mean = torch.tensor(self.cfg["mean"], device=self.dev).view(1, 3, 1, 1)
        self.std = torch.tensor(self.cfg["std"], device=self.dev).view(1, 3, 1, 1)
        self.batch, self.workers = batch, workers
        self._train = None

    def build_copy(self):
        return copy.deepcopy(self.model).eval()

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

    def logits(self, model, x):
        return model({"image": x, "label": torch.zeros(x.shape[0], dtype=torch.long, device=x.device)},
                     inference=True)["cls"]

    def score_split(self, models, split):
        from torch.utils.data import DataLoader
        items = self.Folder(self.images, split).items
        ys, out = [], [[] for _ in models]
        for x, y in DataLoader(self.Images(items, self.size, False), batch_size=self.batch,
                               num_workers=self.workers, pin_memory=True):
            xn = self.norm(x)
            for k, m in enumerate(models):
                out[k].append(self.logits(m, xn).float().softmax(1)[:, 1].cpu())
            ys.append(y)
        return [torch.cat(o).numpy() for o in out], torch.cat(ys).numpy(), len(items)


TARGETS = {"mamba_diloco": MambaTarget, "dfb_effnetb4": DFBTarget}


def _counts(model):
    params = sum(p.numel() for p in model.parameters())
    bufs = sum(b.numel() for b in model.buffers() if b.is_floating_point())
    return params, bufs


class Session:
    def __init__(self, target, **kw):
        _no_grad_guard()
        self.target_name = target
        self.t = TARGETS[target](**kw)
        self.model = self.t.model
        self.dir = RESULTS / target
        self.dir.mkdir(parents=True, exist_ok=True)
        self.budget = Budget(self.dir)
        self.base = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.param_names = {n for n, _ in self.model.named_parameters()}
        self.total_params, self.total_buffers = _counts(self.model)
        self.cur_path = self.dir / "current.pt"
        self.edits_path = self.dir / "edits.jsonl"
        if self.cur_path.exists():
            over = torch.load(self.cur_path, map_location="cpu")
            self._load_overrides(over)
        self._forbid_autograd()

    def _forbid_autograd(self):
        for p in self.model.parameters():
            p.requires_grad_(False)

        def banned(*a, **k):
            raise Banned("autograd/backward is banned in WSP")

        torch.Tensor.backward = banned
        torch.autograd.backward = banned
        torch.autograd.grad = banned
        for name in ("SGD", "Adam", "AdamW", "LBFGS", "RMSprop", "Adagrad"):
            setattr(torch.optim, name, banned)
        torch.linalg.lstsq = banned

    def _load_overrides(self, over):
        sd = self.model.state_dict()
        for k, v in over.items():
            sd[k].copy_(v.to(sd[k].device, sd[k].dtype))

    def state(self, name):
        return self.model.state_dict()[name].detach().cpu().clone()

    def base_state(self, name):
        return self.base[name].clone()

    def names(self):
        return [(k, tuple(v.shape), v.numel(), k in self.param_names) for k, v in self.model.state_dict().items()]

    def module(self, name):
        return self.model.get_submodule(name)

    def _save_current(self):
        sd = self.model.state_dict()
        over = {k: sd[k].detach().cpu().clone() for k in sd if not torch.equal(sd[k].detach().cpu(), self.base[k])}
        torch.save(over, self.cur_path)
        return over

    def edit(self, name, new, hypothesis, reason_detail=None):
        sd = self.model.state_dict()
        old = sd[name].detach().cpu().clone()
        new = new.detach().cpu().to(old.dtype).reshape(old.shape)
        delta_mask = new != old
        n_step = int(delta_mask.sum())
        cum_mask = new != self.base[name]
        n_cum = int(cum_mask.sum())
        sd[name].copy_(new.to(sd[name].device))
        over = self._save_current()
        diff = {}
        if n_step <= 2048:
            idx = delta_mask.nonzero().tolist()
            diff = {"indices": idx, "old": old[delta_mask].tolist(), "new": new[delta_mask].tolist()}
        else:
            d = (new - old).float()
            diff = {"delta_l2": float(d.norm()), "delta_max_abs": float(d.abs().max()),
                    "old_l2": float(old.float().norm()), "new_l2": float(new.float().norm())}
            dfile = self.dir / "diffs"
            dfile.mkdir(exist_ok=True)
            fn = dfile / f"{len(self.edit_log()) + 1:03d}_{name}.pt"
            torch.save({"mask": delta_mask, "old": old[delta_mask], "new": new[delta_mask]}, fn)
            diff["file"] = str(fn.relative_to(self.dir))
        is_param = name in self.param_names
        rec = {"edit": len(self.edit_log()) + 1, "time": now(), "tensor": name,
               "kind": "parameter" if is_param else "buffer", "shape": list(old.shape), "numel": old.numel(),
               "changed_this_edit": n_step, "changed_vs_original": n_cum,
               "pct_of_tensor": 100.0 * n_step / old.numel(),
               "pct_of_total_params": 100.0 * n_step / self.total_params, "hypothesis": hypothesis,
               "detail": reason_detail, "diff": diff, "totals": self.totals(over)}
        with open(self.edits_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        return rec

    def revert(self, name, reason):
        return self.edit(name, self.base[name], f"REVERT: {reason}")

    def edit_log(self):
        if not self.edits_path.exists():
            return []
        return [json.loads(l) for l in self.edits_path.read_text().splitlines() if l.strip()]

    def totals(self, over=None):
        sd = self.model.state_dict()
        cp = cb = 0
        for k in sd:
            if not sd[k].is_floating_point():
                continue
            n = int((sd[k].detach().cpu() != self.base[k]).sum())
            if k in self.param_names:
                cp += n
            else:
                cb += n
        return {"changed_params": cp, "total_params": self.total_params,
                "pct_params": 100.0 * cp / self.total_params, "changed_buffer_values": cb,
                "total_buffer_values": self.total_buffers}

    def _pick(self, n, balanced, reuse):
        if reuse is not None:
            for p in self.budget.data["passes"]:
                if p["pass"] == reuse:
                    return p["indices"]
            raise ValueError(f"no pass {reuse}")
        rng = np.random.default_rng(1000 + self.budget.data["used"])
        if balanced:
            y = self.t.train_labels().numpy()
            a = rng.choice(np.flatnonzero(y == 0), n // 2, replace=False)
            b = rng.choice(np.flatnonzero(y == 1), n - n // 2, replace=False)
            idx = np.concatenate([a, b])
        else:
            idx = rng.choice(self.t.n_train(), n, replace=False)
        return sorted(int(i) for i in idx)

    def practice(self, note, n=64, hooks=(), inputs_of=(), balanced=True, reuse=None, pool=True, bn_batch_stats=False,
                 bn_moments=False):
        if n > 64:
            raise ValueError("practice batches are capped at 64 images")
        idx = self._pick(n, balanced, reuse)
        self.budget.charge("practice", n, note, {"indices": idx, "reuse": reuse})
        x, y, names = self.t.practice_inputs(idx)
        acts, handles = {}, []

        def red(t):
            t = t.detach()
            if pool and t.dim() == 4:
                t = t.mean((2, 3))
            return t.float().cpu()

        for h in hooks:
            handles.append(self.module(h).register_forward_hook(
                lambda m, i, o, h=h: acts.__setitem__(h, red(o if torch.is_tensor(o) else o[0]))))
        for h in inputs_of:
            handles.append(self.module(h).register_forward_hook(
                lambda m, i, o, h=h: acts.__setitem__("in:" + h, red(i[0]))))
        bns = [(n_, m) for n_, m in self.model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        moments = {}
        if bn_moments:
            for n_, m in bns:
                def mh(m, i, o, n_=n_):
                    t = i[0].detach().float()
                    dims = [0, 2, 3] if t.dim() == 4 else [0]
                    cnt = t.numel() // t.shape[1]
                    moments[n_] = (t.sum(dims).cpu(), (t * t).sum(dims).cpu(), cnt)
                handles.append(m.register_forward_hook(mh))
        saved = []
        if bn_batch_stats:
            for n_, m in bns:
                nbt = m.num_batches_tracked.clone() if m.num_batches_tracked is not None else None
                saved.append((m, m.training, m.momentum, nbt))
                m.train()
                m.momentum = 0.0
        try:
            logits = self.t.logits(self.model, x).float().cpu()
        finally:
            for h in handles:
                h.remove()
            for m, tr, mo, nbt in saved:
                m.train(tr)
                m.momentum = mo
                if nbt is not None:
                    m.num_batches_tracked.copy_(nbt)
        return {"logits": logits, "p_fake": logits.softmax(1)[:, 1], "labels": y, "names": names, "indices": idx,
                "acts": acts, "inputs": x.detach().float().cpu() if self.target_name == "mamba_diloco" else None,
                "bn_moments": moments}

    def check_val(self, note):
        rec = self.budget.charge("val", -1, note)
        (p,), y, n = self.t.score_split([self.model], "val" if self.target_name == "mamba_diloco" else "Validation")
        s = summary(y, p)
        s["mean_p_fake_on_real"] = float(p[y == 0].mean())
        s["mean_p_fake_on_fake"] = float(p[y == 1].mean())
        s["pass"] = rec["pass"]
        s["note"] = note
        s["edits_so_far"] = len(self.edit_log())
        s["totals"] = self.totals()
        rec.update({"n": n, "result": {k: s[k] for k in ("accuracy", "auc", "tp", "tn", "fp", "fn")}})
        self.budget.save()
        with open(self.dir / "val_checks.jsonl", "a") as f:
            f.write(json.dumps(s) + "\n")
        return s

    def final(self):
        fpath = self.dir / "final.json"
        if fpath.exists() or self.budget.data.get("final"):
            raise BudgetExceeded("final already run for this target")
        self.budget.data["final"] = {"time": now(), "status": "started"}
        self.budget.save()
        base_model = self.t.build_copy()
        sd = base_model.state_dict()
        for k in sd:
            sd[k].copy_(self.base[k].to(sd[k].device))
        (pb, pe), y, n = self.t.score_split([base_model, self.model], "test" if self.target_name == "mamba_diloco"
                                            else "Test")
        out = {"target": self.target_name, "split": "Test", "n": n, "time": now(),
               "original_weights": summary(y, pb), "edited_weights": summary(y, pe),
               "forward_passes_used": self.budget.data["used"], "val_checks_used": self.budget.data["val_used"],
               "edits": len(self.edit_log()), "totals": self.totals(),
               "note": "single read of Test; original and edited weights scored in the same pass over the data"}
        for k in ("original_weights", "edited_weights"):
            p = pb if k == "original_weights" else pe
            out[k]["mean_p_fake_on_real"] = float(p[y == 0].mean())
            out[k]["mean_p_fake_on_fake"] = float(p[y == 1].mean())
        fpath.write_text(json.dumps(out, indent=1))
        self.budget.data["final"] = {"time": now(), "status": "done"}
        self.budget.save()
        return out


def weight_stats(t):
    t = t.float()
    s = {"shape": list(t.shape), "l2": float(t.norm()), "mean": float(t.mean()), "std": float(t.std()) if t.numel() > 1
         else 0.0, "max_abs": float(t.abs().max())}
    if t.dim() == 2:
        sv = torch.linalg.svdvals(t)
        s["top_sv"] = [float(v) for v in sv[:5]]
        s["eff_rank"] = float((sv.sum() ** 2) / (sv ** 2).sum())
    return s
