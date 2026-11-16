import argparse
import fcntl
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOME = Path.home()
OUT = ROOT / "compare" / "results" / "weight_space"
CACHE = HOME / "wsp_cache"
NAMES = ["xception", "efficientnetb4", "cnnaug_resnet34", "univfd", "effort"]
TRAINING = str(HOME / "sota/DeepfakeBench/training")
WEIGHTS = str(HOME / "sota/dfb_weights")
PYDEPS = str(HOME / "sota/pydeps")
CLIP_PATH = str(HOME / "sota/hf_clip/clip-vit-large-patch14")
FC_PATH = str(HOME / "sota/UnivFD/pretrained_weights/fc_weights.pth")
EFFORT_CKPT = str(HOME / "sota/effort_weights/effort_clip_L14_trainOn_FaceForensic.pth")
KS = [4, 8, 32, 128, 1024]
SEEDS = [0, 1, 2, 3, 4]
ALPHAS = [0.25, 0.5, 0.75]
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEV = torch.device("cuda")
ADAK_KS = [int(v) for v in os.environ.get("WSP_ADAK_KS", "4,8,32,128,1024").split(",")]


@contextmanager
def heavy():
    fh = open(os.path.join(os.environ.get("DFM_SCRATCH", "/tmp"), "gpu_heavy.lock"), "a")
    fcntl.flock(fh, fcntl.LOCK_EX)
    try:
        yield
    finally:
        fcntl.flock(fh, fcntl.LOCK_UN)
        fh.close()


def test_feats(M, test):
    with heavy():
        return M.feats(test, 64)


def data_root():
    h = HOME / "data" / "MegaDataset"
    if h.exists():
        n = sum(len(fs) for _, _, fs in os.walk(h))
        if n == 190335:
            return h
    return Path(os.environ.get("DFM_MEGADATASET", os.path.expanduser("~/data/MegaDataset")))


def list_split(split):
    base = data_root() / split
    out = {}
    for cls in ("Real", "Fake"):
        out[cls] = sorted(str(p) for p in (base / cls).iterdir() if p.suffix.lower() in EXTS)
    return out["Real"], out["Fake"]


def read_one(p):
    img = Image.open(p).convert("RGB")
    if img.size != (256, 256):
        img = img.resize((256, 256), Image.BICUBIC)
    return np.asarray(img)


def load_u8(paths, workers=16):
    arr = np.empty((len(paths), 256, 256, 3), dtype=np.uint8)
    with ThreadPoolExecutor(workers) as ex:
        for i, a in enumerate(ex.map(read_one, paths)):
            arr[i] = a
    return arr


def build_data():
    CACHE.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not (CACHE / "test_u8.npy").exists():
        r, f = list_split("Test")
        paths = r + f
        y = np.array([0] * len(r) + [1] * len(f), dtype=np.int64)
        arr = load_u8(paths)
        np.save(CACHE / "test_y.npy", y)
        np.save(CACHE / "test_u8.npy", arr)
        print("test", arr.shape, round(time.time() - t0), "s", flush=True)
    if not (CACHE / "val_u8.npy").exists():
        r, f = list_split("Validation")
        rng = np.random.default_rng(7)
        pr = [r[i] for i in rng.choice(len(r), 1000, replace=False)]
        pf = [f[i] for i in rng.choice(len(f), 1000, replace=False)]
        np.save(CACHE / "val_y.npy", np.array([0] * 1000 + [1] * 1000, dtype=np.int64))
        np.save(CACHE / "val_u8.npy", load_u8(pr + pf))
        print("val done", flush=True)
    tr = tf = None
    for s in SEEDS:
        if (CACHE / f"pool_u8_{s}.npy").exists() and (CACHE / f"unl_u8_{s}.npy").exists():
            continue
        if tr is None:
            tr, tf = list_split("Train")
            print("train", len(tr), len(tf), flush=True)
        rng = np.random.default_rng(s)
        ri = rng.choice(len(tr), 512, replace=False)
        fi = rng.choice(len(tf), 512, replace=False)
        pool = [tr[i] for i in ri] + [tf[i] for i in fi]
        allp = tr + tf
        ui = rng.choice(len(allp), 1000, replace=False)
        np.save(CACHE / f"pool_y_{s}.npy", np.array([0] * 512 + [1] * 512, dtype=np.int64))
        np.save(CACHE / f"pool_u8_{s}.npy", load_u8(pool))
        np.save(CACHE / f"unl_u8_{s}.npy", load_u8([allp[i] for i in ui]))
        np.save(CACHE / f"unl_y_{s}.npy", np.array([0 if i < len(tr) else 1 for i in ui], dtype=np.int64))
        print("seed", s, "done", flush=True)


def pool_idx(k):
    h = k // 2
    return np.concatenate([np.arange(h), 512 + np.arange(h)])


class Model:
    def __init__(self, name):
        self.name = name
        self.crop = False
        if name in ("xception", "efficientnetb4", "cnnaug_resnet34"):
            from compare.sota_zero_shot import build_dfb
            self.model, cfg, self.load_res = build_dfb(name, TRAINING, WEIGHTS, os.environ.get("DFM_SCRATCH", "/tmp"), PYDEPS)
            self.size, mean, std = cfg["resolution"], cfg["mean"], cfg["std"]
            bb = self.model.backbone
            self.head = {"xception": lambda: bb.last_linear, "efficientnetb4": lambda: bb.last_layer,
                         "cnnaug_resnet34": lambda: bb.fc}[name]()
        elif name == "univfd":
            from compare.sota_zero_shot import UnivFD
            from compare.backbones import CLIP
            self.model, self.load_res = UnivFD(CLIP_PATH, FC_PATH), None
            self.size, self.crop = 224, True
            mean, std = CLIP
            self.head = self.model.fc
        else:
            from transformers import AutoProcessor, CLIPModel, ViTModel, ViTConfig
            from compare.sota_zero_shot import build_effort
            self.model, cfg, self.load_res = build_effort(TRAINING, EFFORT_CKPT, CLIP_PATH, PYDEPS)
            self.size, mean, std = cfg["resolution"], cfg["mean"], cfg["std"]
            self.head = self.model.head
        self.model = self.model.to(DEV).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.mean = torch.tensor(mean, device=DEV).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=DEV).view(1, 3, 1, 1)
        self.two = self.head.out_features == 2
        self.bns = [m for m in self.model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
        self.w_orig = self.head.weight.detach().clone()
        self.b_orig = self.head.bias.detach().clone()

    def prep(self, u8):
        x = torch.from_numpy(np.ascontiguousarray(u8)).to(DEV).permute(0, 3, 1, 2).float() / 255.0
        if self.crop:
            o = (256 - self.size) // 2
            x = x[:, :, o:o + self.size, o:o + self.size]
        elif self.size != 256:
            x = F.interpolate(x, size=(self.size, self.size), mode="bicubic", antialias=True, align_corners=False)
        return (x - self.mean) / self.std

    def trunk(self, x):
        n = self.name
        if n in ("xception", "efficientnetb4", "cnnaug_resnet34"):
            f = self.model.features({"image": x})
            if n == "xception":
                f = F.relu(f)
            return F.adaptive_avg_pool2d(f, 1).flatten(1)
        if n == "univfd":
            return self.model.clip(pixel_values=x).image_embeds
        return self.model.features({"image": x})

    def full_score(self, x):
        n = self.name
        if n == "univfd":
            return self.model(x)[:, 0].double()
        lab = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        z = self.model({"image": x, "label": lab}, inference=True)["cls"].double()
        return z[:, 1] - z[:, 0]

    @torch.no_grad()
    def feats(self, u8, bs=32):
        out = []
        for i in range(0, len(u8), bs):
            out.append(self.trunk(self.prep(u8[i:i + bs])).float())
        return torch.cat(out)

    @torch.no_grad()
    def scores_full(self, u8, bs=32):
        return torch.cat([self.full_score(self.prep(u8[i:i + bs])) for i in range(0, len(u8), bs)])

    def diff_head(self):
        W, b = self.w_orig.double(), self.b_orig.double()
        if self.two:
            return W[1] - W[0], b[1] - b[0]
        return W[0], b[0]

    @torch.no_grad()
    def set_head(self, w, b):
        w, b = w.to(self.head.weight.dtype), b.to(self.head.weight.dtype)
        if self.two:
            self.head.weight[0] = -w / 2
            self.head.weight[1] = w / 2
            self.head.bias[0] = -b / 2
            self.head.bias[1] = b / 2
        else:
            self.head.weight[0] = w
            self.head.bias[0] = b

    @torch.no_grad()
    def reset_head(self):
        self.head.weight.copy_(self.w_orig)
        self.head.bias.copy_(self.b_orig)

    def bn_state(self):
        return [(m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone(), m.momentum)
                for m in self.bns]

    def bn_restore(self, st):
        with torch.no_grad():
            for m, (a, v, n, mo) in zip(self.bns, st):
                m.running_mean.copy_(a)
                m.running_var.copy_(v)
                m.num_batches_tracked.copy_(n)
                m.momentum = mo
        self.model.eval()

    @torch.no_grad()
    def adabn(self, u8, bs=64):
        for m in self.bns:
            m.reset_running_stats()
            m.momentum = None
            m.train()
        for i in range(0, len(u8), bs):
            self.trunk(self.prep(u8[i:i + bs]))
        self.model.eval()

    def edit_counts(self):
        D = self.head.in_features
        head = (2 * D + 2) if self.two else (D + 1)
        free = D + 1
        bn = sum(m.running_mean.numel() + m.running_var.numel() for m in self.bns)
        return {"feature_dim": D, "head_numbers_written": head, "head_free_parameters": free,
                "bn_numbers_rewritten": bn, "total_params": sum(p.numel() for p in self.model.parameters())}


def auc_t(y, s):
    s = s.double()
    u, inv, cnt = torch.unique(s, return_inverse=True, return_counts=True)
    start = torch.cumsum(cnt, 0) - cnt
    r = (start + (cnt + 1) / 2.0)[inv]
    n1 = (y == 1).sum().double()
    n0 = (y == 0).sum().double()
    return float(((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)).item())


def metrics(y, s):
    return float(((s > 0).long() == y).double().mean().item()), auc_t(y, s)


def cf_ridge(f, y, rel):
    f = f.double()
    t = y.double() * 2 - 1
    mu, tm = f.mean(0), t.mean()
    X, tc = f - mu, t - tm
    K = X @ X.T
    lam = rel * torch.trace(K) / len(f)
    a = torch.linalg.solve(K + lam * torch.eye(len(f), device=f.device, dtype=f.dtype), tc)
    w = X.T @ a
    return w, tm - mu @ w


def cf_centroid(f, y):
    f = f.double()
    m1, m0 = f[y == 1].mean(0), f[y == 0].mean(0)
    return m1 - m0, -(m1 @ m1 - m0 @ m0) / 2


def cf_lda(f, y, s):
    f = f.double()
    m1, m0 = f[y == 1].mean(0), f[y == 0].mean(0)
    X = f - torch.where((y == 1)[:, None], m1[None], m0[None])
    S = X.T @ X / len(f)
    D = f.shape[1]
    Sig = (1 - s) * S + s * (torch.trace(S) / D) * torch.eye(D, device=f.device, dtype=f.dtype)
    w = torch.linalg.solve(Sig, m1 - m0)
    return w, -w @ (m1 + m0) / 2


def bias_shift(fk, yk, w0, b0, optimal=False):
    sc = fk.double() @ w0 + b0
    t = (sc[yk == 1].mean() + sc[yk == 0].mean()) / 2
    if optimal:
        ss = torch.sort(sc).values
        cand = (ss[1:] + ss[:-1]) / 2
        acc = ((sc[None, :] > cand[:, None]).long() == yk[None, :]).double().mean(1)
        best = acc.max()
        ok = cand[acc == best]
        t = ok[torch.argmin((ok - t).abs())]
    return w0, b0 - t


def head_set(fk, yk, w0, b0, hp, wise=True):
    h = {"bias": bias_shift(fk, yk, w0, b0), "bias_opt": bias_shift(fk, yk, w0, b0, True),
         "ridge": cf_ridge(fk, yk, hp["ridge_rel"]), "centroid": cf_centroid(fk, yk),
         "lda": cf_lda(fk, yk, hp["lda_shrink"])}
    if wise:
        for base in ("ridge", "lda"):
            for a in ALPHAS:
                w, b = h[base]
                h[f"wise_{base}_{a}"] = ((1 - a) * w0 + a * w, (1 - a) * b0 + a * b)
    return h


def probe(name):
    from collections import OrderedDict
    M = Model(name)
    y = np.load(CACHE / "test_y.npy")
    u8 = np.load(CACHE / "test_u8.npy", mmap_mode="r")
    idx = np.sort(np.random.default_rng(0).choice(len(y), 5, replace=False))
    imgs = np.asarray(u8[idx])
    calls = OrderedDict()

    def shp(o):
        if isinstance(o, torch.Tensor):
            return list(o.shape)
        if isinstance(o, dict):
            return {k: shp(v) for k, v in o.items() if v is not None and shp(v) is not None}
        if isinstance(o, (tuple, list)):
            r = [shp(v) for v in o if v is not None]
            return [v for v in r if v is not None] or None
        return None

    hooks = []
    for n, m in M.model.named_modules():
        if n == "":
            continue

        def mk(n):
            def h(mod, i, o):
                if n not in calls:
                    calls[n] = {"in": shp(i), "out": shp(o)}
            return h
        hooks.append(m.register_forward_hook(mk(n)))
    with torch.no_grad():
        x = M.prep(imgs)
        feat = M.trunk(x)
        if name in ("xception", "efficientnetb4", "cnnaug_resnet34", "effort"):
            M.model({"image": x, "label": torch.zeros(5, dtype=torch.long, device=DEV)}, inference=True)
        else:
            M.model(x)
    for h in hooks:
        h.remove()
    mods = dict(M.model.named_modules())
    full = []
    for n, m in mods.items():
        if n == "":
            continue
        rec = {"name": n, "type": type(m).__name__, "called": n in calls,
               "params": {k: list(p.shape) for k, p in m.named_parameters(recurse=False)},
               "n_params_total": sum(p.numel() for p in m.parameters())}
        if n in calls:
            rec["in"], rec["out"] = calls[n]["in"], calls[n]["out"]
        full.append(rec)
    expand = {"univfd": {"clip", "clip.vision_model", "clip.vision_model.encoder"},
              "effort": {"backbone", "backbone.encoder"},
              "cnnaug_resnet34": {"backbone.resnet"}}.get(name, set())
    children = {}
    for n in mods:
        if n:
            children.setdefault(n.rsplit(".", 1)[0] if "." in n else "", []).append(n)
    byname = {r["name"]: r for r in full}
    top = []

    def walk(n):
        for c in children.get(n, []):
            if c in byname and byname[c]["called"] and c not in expand and "loss" not in c:
                top.append(byname[c])
            elif c in byname and "loss" not in c:
                walk(c)
    walk("")
    groups = []
    for r in top:
        sig = (json.dumps(r.get("in")), json.dumps(r.get("out")), r["type"], r["n_params_total"],
               json.dumps(r["params"]))
        if groups and groups[-1]["sig"] == sig:
            groups[-1]["names"].append(r["name"])
        else:
            groups.append({"sig": sig, "names": [r["name"]], "rec": r})
    compact = []
    for g in groups:
        r = g["rec"]
        nm = g["names"][0] if len(g["names"]) == 1 else f"{g['names'][0]} .. {g['names'][-1]} (x{len(g['names'])})"
        compact.append({"layers": nm, "type": r["type"], "input": r.get("in"), "output": r.get("out"),
                        "param_shapes": r["params"], "params_each": r["n_params_total"]})
    hs = M.head.weight.shape
    rec = {"model": name, "probe_images": 5, "probe_indices": idx.tolist(), "input_resolution": M.size,
           "input_shape": [5, 3, M.size, M.size], "penultimate_feature_shape": list(feat.shape),
           "penultimate_dim": int(feat.shape[1]), "head_W_shape": list(hs), "head_b_shape": list(M.head.bias.shape),
           "n_batchnorm_layers": len(M.bns), "total_params": M.edit_counts()["total_params"],
           "checkpoint_missing_keys": None if M.load_res is None else len(M.load_res.missing_keys),
           "checkpoint_unexpected_keys": None if M.load_res is None else len(M.load_res.unexpected_keys),
           "compact": compact, "layers_full": full}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"probe_{name}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(name, "D=", rec["penultimate_dim"], "W", rec["head_W_shape"], "compact rows", len(compact), flush=True)


def cache_feats(name):
    M = Model(name)
    test = np.load(CACHE / "test_u8.npy", mmap_mode="r")
    val = np.load(CACHE / "val_u8.npy")
    t0 = time.time()
    ft = M.feats(test, 64)
    fv = M.feats(val, 64)
    np.save(CACHE / f"{name}_test_f.npy", ft.cpu().numpy())
    np.save(CACHE / f"{name}_val_f.npy", fv.cpu().numpy())
    print(name, "cached", tuple(ft.shape), round(time.time() - t0), "s", flush=True)
    del M, ft, fv
    torch.cuda.empty_cache()


def tune(name):
    fv = torch.from_numpy(np.load(CACHE / f"{name}_val_f.npy")).to(DEV)
    yv = torch.from_numpy(np.load(CACHE / "val_y.npy")).to(DEV)
    M = Model(name)
    w0, b0 = M.diff_head()
    ridge_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    lda_grid = [0.01, 0.1, 0.3, 0.6, 0.9]
    res = {"ridge": {str(g): [] for g in ridge_grid}, "lda": {str(g): [] for g in lda_grid}}
    for kk in (8, 32, 128):
        for d in range(10):
            rng = np.random.default_rng(1000 + d)
            ii = np.concatenate([rng.choice(1000, kk // 2, replace=False), 1000 + rng.choice(1000, kk // 2, replace=False)])
            ii = torch.from_numpy(ii).to(DEV)
            fk, yk = fv[ii], yv[ii]
            for g in ridge_grid:
                w, b = cf_ridge(fk, yk, g)
                res["ridge"][str(g)].append(metrics(yv, fv.double() @ w + b)[0])
            for g in lda_grid:
                w, b = cf_lda(fk, yk, g)
                res["lda"][str(g)].append(metrics(yv, fv.double() @ w + b)[0])
    mean = {m: {g: float(np.mean(v)) for g, v in d.items()} for m, d in res.items()}
    hp = {"ridge_rel": float(max(mean["ridge"], key=mean["ridge"].get)),
          "lda_shrink": float(max(mean["lda"], key=mean["lda"].get))}
    rec = {"model": name, "selected": hp, "val_mean_acc": mean, "protocol": "10 draws each of k=8,32,128 from the 2000-image Validation subset, original trunk"}
    (OUT / f"tune_{name}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return hp


def run_rows(name, stages):
    OUT.mkdir(parents=True, exist_ok=True)
    M = Model(name)
    hp = tune(name)
    print(name, "hp", hp, flush=True)
    yt = torch.from_numpy(np.load(CACHE / "test_y.npy")).to(DEV)
    test = np.load(CACHE / "test_u8.npy", mmap_mode="r")
    Ft0 = torch.from_numpy(np.load(CACHE / f"{name}_test_f.npy")).to(DEV)
    w0, b0 = M.diff_head()
    rows = []

    def add(ctx, meth, k, seed, Ft, w, b):
        a, u = metrics(yt, Ft.double() @ w + b)
        rows.append({"model": name, "method": f"{ctx}/{meth}", "k": k, "seed": seed, "acc": a, "auc": u})

    if "base" in stages:
        add("base", "orig", 0, 0, Ft0, w0, b0)
        for s in SEEDS:
            pool = np.load(CACHE / f"pool_u8_{s}.npy")
            py = torch.from_numpy(np.load(CACHE / f"pool_y_{s}.npy")).to(DEV)
            Fp = M.feats(pool)
            for k in KS:
                ii = torch.from_numpy(pool_idx(k)).to(DEV)
                for meth, (w, b) in head_set(Fp[ii], py[ii], w0, b0, hp).items():
                    add("base", meth, k, s, Ft0, w, b)
        save_rows(name, "base", rows)
        rows = []

    if "ada1000" in stages and M.bns:
        st = M.bn_state()
        for s in SEEDS:
            pool = np.load(CACHE / f"pool_u8_{s}.npy")
            py = torch.from_numpy(np.load(CACHE / f"pool_y_{s}.npy")).to(DEV)
            unl = np.load(CACHE / f"unl_u8_{s}.npy")
            M.adabn(unl)
            Ft = test_feats(M, test)
            Fp = M.feats(pool)
            add("adabn1000", "orig", 0, s, Ft, w0, b0)
            for k in KS:
                ii = torch.from_numpy(pool_idx(k)).to(DEV)
                for meth, (w, b) in head_set(Fp[ii], py[ii], w0, b0, hp, wise=False).items():
                    add("adabn1000", meth, k, s, Ft, w, b)
            M.bn_restore(st)
            print(name, "ada1000 seed", s, flush=True)
        save_rows(name, "ada1000", rows)
        rows = []

    if "adak" in stages and M.bns:
        st = M.bn_state()
        for s in SEEDS:
            pool = np.load(CACHE / f"pool_u8_{s}.npy")
            py = torch.from_numpy(np.load(CACHE / f"pool_y_{s}.npy")).to(DEV)
            for k in ADAK_KS:
                pi = pool_idx(k)
                M.adabn(pool[pi])
                Ft = test_feats(M, test)
                Fk = M.feats(pool[pi])
                yk = py[torch.from_numpy(pi).to(DEV)]
                add("adabn_k", "orig", k, s, Ft, w0, b0)
                for meth, (w, b) in head_set(Fk, yk, w0, b0, hp, wise=False).items():
                    add("adabn_k", meth, k, s, Ft, w, b)
                M.bn_restore(st)
                print(name, "adak seed", s, "k", k, flush=True)
            save_rows(name, "adak", rows)
        rows = []

    if "verify" in stages:
        verify(M, name, hp, test, Ft0, w0, b0)
    del M, Ft0
    torch.cuda.empty_cache()


def save_rows(name, stage, rows):
    (OUT / f"rows_{name}_{stage}.json").write_text(json.dumps(rows))


def verify(M, name, hp, test, Ft0, w0, b0):
    pool = np.load(CACHE / "pool_u8_0.npy")
    py = torch.from_numpy(np.load(CACHE / "pool_y_0.npy")).to(DEV)
    n = 256
    sub = test[::len(test) // n][:n]
    ii = pool_idx(32)
    sel = torch.from_numpy(ii).to(DEV)
    Fp = M.feats(pool)
    hs = head_set(Fp[sel], py[sel], w0, b0, hp)
    recs = {}
    sidx = np.arange(0, len(test), len(test) // n)[:n]
    Fs = Ft0[torch.from_numpy(sidx).to(DEV)].double()
    for meth, (w, b) in hs.items():
        M.set_head(w, b)
        got = M.scores_full(sub)
        want = Fs @ w + b
        recs[meth] = {"max_abs_diff": float((got - want).abs().max()), "max_abs_score": float(want.abs().max()),
                      "sign_agreement": float(((got > 0) == (want > 0)).double().mean())}
        M.reset_head()
    M.set_head(w0, b0)
    recs["original_rewritten"] = {"max_abs_diff": float((M.scores_full(sub) - (Fs @ w0 + b0)).abs().max())}
    M.reset_head()
    out = {"model": name, "setting": "seed 0, k=32, 256 Test images through the full network after set_head", "heads": recs}
    if M.bns:
        st = M.bn_state()
        M.adabn(pool[ii])
        Fk = M.feats(pool[ii])
        w, b = cf_ridge(Fk, py[sel], hp["ridge_rel"])
        Fa = M.feats(sub).double()
        M.set_head(w, b)
        got = M.scores_full(sub)
        out["adabn_plus_ridge"] = {"max_abs_diff": float((got - (Fa @ w + b)).abs().max()),
                                  "max_abs_score": float((Fa @ w + b).abs().max())}
        M.reset_head()
        M.bn_restore(st)
    (OUT / f"verify_{name}.json").write_text(json.dumps(out, indent=1))
    print(name, "verify", {k: round(v["max_abs_diff"], 6) for k, v in recs.items()}, flush=True)


def fmt(m, s):
    return f"{m:.3f}±{s:.3f}"


def report():
    rows = []
    for p in sorted(OUT.glob("rows_*.json")):
        rows += json.loads(p.read_text())
    agg = {}
    for r in rows:
        agg.setdefault((r["model"], r["method"], r["k"]), []).append(r)
    summ = []
    for (m, meth, k), rs in sorted(agg.items()):
        a = np.array([r["acc"] for r in rs])
        u = np.array([r["auc"] for r in rs])
        sd = (lambda v: float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        summ.append({"model": m, "method": meth, "k": k, "n_seeds": len(rs), "acc_mean": float(a.mean()),
                     "acc_std": sd(a), "auc_mean": float(u.mean()), "auc_std": sd(u)})
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    lines = []
    models = [n for n in NAMES if any(s["model"] == n for s in summ)]
    for m in models:
        ms = [s for s in summ if s["model"] == m]
        base = next((s for s in ms if s["method"] == "base/orig" and s["k"] == 0), None)
        lines.append(f"### {m}\n")
        if base:
            lines.append(f"Original (k=0): acc {base['acc_mean']:.3f}, AUC {base['auc_mean']:.3f}\n")
        meths = []
        for s in ms:
            if s["method"] not in meths and s["k"] != 0 or (s["method"] not in meths and s["method"] != "base/orig"):
                meths.append(s["method"])
        for metric, label in (("acc", "Test accuracy"), ("auc", "Test AUC")):
            lines.append(f"{label}, mean±std over seeds\n")
            lines.append("| method | " + " | ".join(f"k={k}" for k in KS) + " |")
            lines.append("|---|" + "---|" * len(KS))
            for meth in meths:
                cells = []
                for k in KS:
                    c = next((s for s in ms if s["method"] == meth and s["k"] == k), None)
                    if c is None:
                        c = next((s for s in ms if s["method"] == meth and s["k"] == 0), None)
                    cells.append(fmt(c[f"{metric}_mean"], c[f"{metric}_std"]) if c else "")
                lines.append(f"| {meth} | " + " | ".join(cells) + " |")
            lines.append("")
    best = ["### Best accuracy per model\n", "| model | k | best method (acc) | best method (AUC) |", "|---|---|---|---|"]
    for m in models:
        for k in (4, 1024):
            c = [s for s in summ if s["model"] == m and s["k"] == k]
            if not c:
                continue
            ba = max(c, key=lambda s: s["acc_mean"])
            bu = max(c, key=lambda s: s["auc_mean"])
            best.append(f"| {m} | {k} | {ba['method']} {fmt(ba['acc_mean'], ba['acc_std'])} | {bu['method']} {fmt(bu['auc_mean'], bu['auc_std'])} |")
    (OUT / "tables.md").write_text("\n".join(best + [""] + lines))
    probe_lines = []
    for m in models:
        p = OUT / f"probe_{m}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        probe_lines.append(f"### {m}\n")
        probe_lines.append(f"input {r['input_shape']}, penultimate feature {r['penultimate_feature_shape']}, head W {r['head_W_shape']}, b {r['head_b_shape']}, BatchNorm layers {r['n_batchnorm_layers']}, params {r['total_params']:,}\n")
        probe_lines.append("| layers | type | input | output | own params | params each |")
        probe_lines.append("|---|---|---|---|---|---|")
        for c in r["compact"]:
            probe_lines.append(f"| {c['layers']} | {c['type']} | {json.dumps(c['input'])} | {json.dumps(c['output'])} | {json.dumps(c['param_shapes'])} | {c['params_each']:,} |")
        probe_lines.append("")
    (OUT / "probe_tables.md").write_text("\n".join(probe_lines))
    print("report written", len(summ), "summary rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["data", "probe", "cache", "run", "report"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--stages", default="base,ada1000,adak,verify")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = False
    if a.cmd == "data":
        build_data()
    elif a.cmd == "probe":
        probe(a.model)
    elif a.cmd == "cache":
        cache_feats(a.model)
    elif a.cmd == "run":
        run_rows(a.model, set(a.stages.split(",")))
    else:
        report()


if __name__ == "__main__":
    main()
