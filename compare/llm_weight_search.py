import argparse, glob, json, os, sys
import numpy as np
from scipy.stats import rankdata

ROOT = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "llm_weight_space")
CLIP = 12.0
SEED = 0


def auc(y, s):
    r = rankdata(s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def acc(y, logit):
    return float(((logit > 0).astype(int) == y).mean())


def zof(p):
    p = np.clip(p.astype(np.float64), 1e-12, 1 - 1e-12)
    return np.clip(np.log(p / (1 - p)), -CLIP, CLIP)


def find_files(split):
    found = {}
    pats = [os.path.join(RES, "zero_shot", "probs", "*_%s.npz" % split),
            os.path.join(RES, "backbones", "**", "*_%s.npz" % split)]
    for pat in pats:
        for f in glob.glob(pat, recursive=True):
            name = os.path.basename(f)[: -len("_%s.npz" % split)]
            found[name] = f
    return found


def load_split(split, names):
    files = find_files(split)
    data = {}
    for n in names:
        d = np.load(files[n], allow_pickle=True)
        data[n] = (d["p"], d["y"].astype(int), [str(x) for x in d["names"]])
    common = None
    for n in names:
        s = set(data[n][2])
        common = s if common is None else common & s
    order = sorted(common)
    ref = None
    Z = []
    for n in names:
        p, y, nm = data[n]
        idx = {k: i for i, k in enumerate(nm)}
        sel = [idx[k] for k in order]
        yy = y[sel]
        if ref is None:
            ref = yy
        elif not np.array_equal(ref, yy):
            sys.exit("label mismatch in %s %s" % (n, split))
        Z.append(zof(p[sel]))
    return np.stack(Z, 1), ref, order


def state_path():
    return os.path.join(OUT, "state.json")


def cmd_init(a):
    os.makedirs(OUT, exist_ok=True)
    v, t = find_files("Validation"), find_files("Test")
    names = sorted(set(v) & set(t))
    if a.detectors:
        names = [n for n in a.detectors.split(",") if n in names]
    Z, y, order = load_split("Validation", names)
    rng = np.random.RandomState(SEED)
    opt = np.zeros(len(y), bool)
    for c in (0, 1):
        ix = np.where(y == c)[0]
        rng.shuffle(ix)
        opt[ix[: len(ix) // 2]] = True
    json.dump({"detectors": names, "n_val": int(len(y)), "n_opt": int(opt.sum()),
               "opt_names": [order[i] for i in np.where(opt)[0]]}, open(state_path(), "w"))
    print("detectors", names)
    print("val", len(y), "opt", int(opt.sum()), "hold", int((~opt).sum()), "fake rate", y.mean())
    for i, n in enumerate(names):
        print(n, "val auc", round(auc(y, Z[:, i]), 4), "acc", round(acc(y, Z[:, i]), 4))


def get_state():
    return json.load(open(state_path()))


def val_views():
    st = get_state()
    names = st["detectors"]
    Z, y, order = load_split("Validation", names)
    opt_set = set(st["opt_names"])
    opt = np.array([o in opt_set for o in order])
    return names, Z[opt], y[opt], Z[~opt], y[~opt]


def logit_of(prm, names, Z):
    w = np.array([prm.get("w", {}).get(n, 0.0) for n in names])
    c = np.array([prm.get("c", {}).get(n, 0.0) for n in names])
    return (Z + c) @ w + prm.get("b", 0.0)


def nparams(prm):
    return len(prm.get("w", {})) + len(prm.get("c", {})) + (1 if "b" in prm else 0)


def score(prm, names, Zo, yo, Zh, yh):
    lo, lh = logit_of(prm, names, Zo), logit_of(prm, names, Zh)
    return {"opt_auc": auc(yo, lo), "opt_acc": acc(yo, lo), "hold_auc": auc(yh, lh), "hold_acc": acc(yh, lh)}


def cmd_eval(a):
    names, Zo, yo, Zh, yh = val_views()
    spec = json.load(open(a.params))
    rnd = spec.get("round", 0)
    props = spec["proposals"] if isinstance(spec, dict) else spec
    path = os.path.join(OUT, a.history)
    for i, prm in enumerate(props):
        if nparams(prm) > 20:
            sys.exit("more than 20 numbers")
        s = score(prm, names, Zo, yo, Zh, yh)
        rec = {"round": rnd, "idx": i, "params": prm, **s}
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("r%s #%d opt_auc=%.4f opt_acc=%.4f hold_auc=%.4f hold_acc=%.4f b=%s w=%s c=%s" % (
            rnd, i, s["opt_auc"], s["opt_acc"], s["hold_auc"], s["hold_acc"], prm.get("b"), prm.get("w"), prm.get("c", {})))


def read_hist(fn):
    p = os.path.join(OUT, fn)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def fit_backprop(Zo, yo, steps=3000, lr=0.02):
    import torch
    torch.manual_seed(SEED)
    X = torch.tensor(Zo, dtype=torch.float32)
    Y = torch.tensor(yo, dtype=torch.float32)
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(X @ w + b, Y)
        loss.backward()
        opt.step()
    return w.detach().numpy().astype(float), float(b.item())


def rand_params(rng, names):
    return {"w": {n: float(rng.uniform(-0.5, 1.5)) for n in names}, "b": float(rng.uniform(-3, 3))}


def cmd_baselines(a):
    names, Zo, yo, Zh, yh = val_views()
    n_llm = len(read_hist("history.jsonl"))
    n = a.n if a.n else n_llm
    path = os.path.join(OUT, "random_history.jsonl")
    if os.path.exists(path):
        os.remove(path)
    rng = np.random.RandomState(SEED + 1)
    for i in range(n):
        prm = rand_params(rng, names)
        s = score(prm, names, Zo, yo, Zh, yh)
        with open(path, "a") as f:
            f.write(json.dumps({"idx": i, "params": prm, **s}) + "\n")
    rh = read_hist("random_history.jsonl")
    best = max(rh, key=lambda r: r["hold_auc"])
    print("random evals", n, "best hold auc", best["hold_auc"], "opt auc", best["opt_auc"])
    sing = [(auc(yo, Zo[:, i]), n_) for i, n_ in enumerate(names)]
    for s_, n_ in sorted(sing, reverse=True):
        print("single", n_, "opt auc", round(s_, 4))
    w, b = fit_backprop(Zo, yo)
    prm = {"w": {n_: float(x) for n_, x in zip(names, w)}, "b": b}
    print("backprop", score(prm, names, Zo, yo, Zh, yh), prm)
    eq = {"w": {n_: 1.0 / len(names) for n_ in names}, "b": 0.0}
    print("equal", score(eq, names, Zo, yo, Zh, yh))


def cmd_final(a):
    fp = os.path.join(OUT, "final.json")
    if os.path.exists(fp) and not a.force:
        sys.exit("final.json exists; Test already scored")
    names, Zo, yo, Zh, yh = val_views()
    Zt, yt, _ = load_split("Test", names)
    h = read_hist("history.jsonl")
    rh = read_hist("random_history.jsonl")
    out = {"detectors": names, "n_test": int(len(yt)), "threshold": "combined logit > 0 (p > 0.5)"}

    def rec(label, prm, nparam, nevals, extra=None):
        lt = logit_of(prm, names, Zt)
        s = score(prm, names, Zo, yo, Zh, yh)
        out[label] = {"params": prm, "n_params": nparam, "n_evals": nevals, "opt_auc": s["opt_auc"],
                      "hold_auc": s["hold_auc"], "test_auc": auc(yt, lt), "test_acc": acc(yt, lt), **(extra or {})}

    best = max(h, key=lambda r: r["hold_auc"])
    rec("llm_opro", best["params"], nparams(best["params"]), len(h), {"round": best["round"], "idx": best["idx"]})
    rbest = max(rh, key=lambda r: r["hold_auc"])
    rec("random_search", rbest["params"], nparams(rbest["params"]), len(rh))
    sa = [auc(yo, Zo[:, i]) for i in range(len(names))]
    i = int(np.argmax(sa))
    rec("best_single", {"w": {names[i]: 1.0}, "b": 0.0}, 0, len(names), {"detector": names[i]})
    rec("equal_weight", {"w": {n: 1.0 / len(names) for n in names}, "b": 0.0}, 0, 1)
    w, b = fit_backprop(Zo, yo)
    rec("backprop_logreg", {"w": {n: float(x) for n, x in zip(names, w)}, "b": b}, len(names) + 1, 3000)
    per = {}
    for j, n in enumerate(names):
        per[n] = {"test_auc": auc(yt, Zt[:, j]), "test_acc": acc(yt, Zt[:, j]), "opt_auc": sa[j]}
    out["per_detector"] = per
    json.dump(out, open(fp, "w"), indent=1)
    for k, v in out.items():
        if isinstance(v, dict) and "test_auc" in v and "n_params" in v:
            print(k, v["n_params"], v["n_evals"], round(v["hold_auc"], 4), round(v["test_acc"], 4), round(v["test_auc"], 4))


def cmd_show(a):
    h = read_hist("history.jsonl")
    h.sort(key=lambda r: -r["opt_auc"])
    for r in h[: a.top]:
        print(r["round"], r["idx"], round(r["opt_auc"], 4), round(r["opt_acc"], 4), round(r["hold_auc"], 4), r["params"])


ap = argparse.ArgumentParser()
sp = ap.add_subparsers(dest="cmd", required=True)
s = sp.add_parser("init")
s.add_argument("--detectors", default="")
s.set_defaults(f=cmd_init)
s = sp.add_parser("eval")
s.add_argument("--params", required=True)
s.add_argument("--history", default="history.jsonl")
s.set_defaults(f=cmd_eval)
s = sp.add_parser("baselines")
s.add_argument("--n", type=int, default=0)
s.set_defaults(f=cmd_baselines)
s = sp.add_parser("final")
s.add_argument("--force", action="store_true")
s.set_defaults(f=cmd_final)
s = sp.add_parser("show")
s.add_argument("--top", type=int, default=10)
s.set_defaults(f=cmd_show)
a = ap.parse_args()
a.f(a)
