import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

ap = argparse.ArgumentParser(description="64-bit dHash of every image under <root>/{Train,Validation,Test}/{Real,Fake}; count Test images within a Hamming distance of any Train image and re-score predictions without them.")
ap.add_argument("root")
ap.add_argument("--cache", default=os.path.expanduser("~/work/dhash_cache.npz"))
ap.add_argument("--out", default="results/dataset_duplicates/near_dup.json")
ap.add_argument("--preds", default=None)
ap.add_argument("--names", default=None)
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--threshold", type=float, default=0.5)
args = ap.parse_args()


def dhash(path):
    try:
        im = Image.open(path)
        im.draft("L", (64, 64))
        g = np.asarray(im.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
        bits = (g[:, 1:] > g[:, :-1]).flatten()
        return int(np.packbits(bits).view(">u8")[0])
    except Exception:
        return 0xFFFFFFFFFFFFFFFF


def listing():
    items = []
    for sp in ("Train", "Validation", "Test"):
        for c in ("Real", "Fake"):
            d = os.path.join(args.root, sp, c)
            for f in sorted(os.listdir(d)):
                items.append((sp, c, f))
    return items


items = listing()
if os.path.exists(args.cache):
    z = np.load(args.cache, allow_pickle=True)
    hashes = z["hashes"]
    assert len(hashes) == len(items)
else:
    paths = [os.path.join(args.root, sp, c, f) for sp, c, f in items]
    with ProcessPoolExecutor(args.workers) as ex:
        hashes = np.array(list(ex.map(dhash, paths, chunksize=256)), dtype=np.uint64)
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    np.savez(args.cache, hashes=hashes)

sp_arr = np.array([i[0] for i in items])
cl_arr = np.array([i[1] for i in items])
name_arr = np.array([f"{i[0]}/{i[1]}/{i[2]}" for i in items])
valid = hashes != np.uint64(0xFFFFFFFFFFFFFFFF)
print("unreadable", int((~valid).sum()), flush=True)


def min_dist(q, ref):
    best = np.full(len(q), 65, dtype=np.int32)
    arg = np.zeros(len(q), dtype=np.int64)
    for i in range(0, len(q), 256):
        x = q[i:i + 256, None] ^ ref[None, :]
        d = np.bitwise_count(x).astype(np.int32)
        a = d.argmin(1)
        best[i:i + 256] = d[np.arange(len(a)), a]
        arg[i:i + 256] = a
    return best, arg


test_idx = np.where((sp_arr == "Test") & valid)[0]
res = {"files": len(items), "unreadable": int((~valid).sum()), "hash": "dHash 64-bit (9x8 grayscale, LANCZOS)", "test_images": int(len(test_idx))}
for ref_name, ref_splits in (("train", ("Train",)), ("train_val", ("Train", "Validation"))):
    ref_idx = np.where(np.isin(sp_arr, ref_splits) & valid)[0]
    dist, arg = min_dist(hashes[test_idx], hashes[ref_idx])
    r = {"reference_images": int(len(ref_idx)), "histogram_min_distance": {str(k): int((dist == k).sum()) for k in range(0, 9)}}
    for t in (0, 2, 4, 8):
        sel = dist <= t
        same = int((cl_arr[test_idx][sel] == cl_arr[ref_idx][arg][sel]).sum())
        r[f"within_{t}"] = {"test_images": int(sel.sum()), "same_label_as_nearest": same, "different_label_as_nearest": int(sel.sum()) - same}
    res[ref_name] = r
    if ref_name == "train":
        dist_train, arg_train, ref_train = dist, arg, ref_idx
    else:
        dist_tv = dist

if args.preds and args.names:
    p = np.load(args.preds)
    names = json.load(open(args.names))
    pos = {n: i for i, n in enumerate(names)}
    name_to_global = {n: i for i, n in enumerate(name_arr)}
    y = np.array([1 if "/Fake/" in n else 0 for n in names])
    pred = (p >= args.threshold).astype(int)
    full_acc = float((pred == y).mean())

    def roc_auc_score(y, p):
        order = np.argsort(p, kind="mergesort")
        ps = p[order]
        _, inv, cnt = np.unique(ps, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt)
        ranks_sorted = (csum - (cnt - 1) / 2.0)[inv]
        ranks = np.empty(len(p))
        ranks[order] = ranks_sorted
        n1 = y.sum()
        n0 = len(y) - n1
        return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

    out = {"threshold": args.threshold, "all": {"n": len(y), "accuracy": full_acc, "auc": float(roc_auc_score(y, p))}}
    near = {}
    for ref_name, dist in (("train", dist_train), ("train_val", dist_tv)):
        dmap = {name_arr[g]: int(d) for g, d in zip(test_idx, dist)}
        for t in (0, 4, 8):
            keep = np.array([dmap.get(n, 65) > t for n in names])
            if keep.all():
                out[f"{ref_name}_without_within_{t}"] = {"removed": 0, "note": "nothing to remove"}
                continue
            out[f"{ref_name}_without_within_{t}"] = {"removed": int((~keep).sum()), "n": int(keep.sum()),
                                                      "accuracy": float((pred[keep] == y[keep]).mean()),
                                                      "auc": float(roc_auc_score(y[keep], p[keep])),
                                                      "accuracy_on_removed": float((pred[~keep] == y[~keep]).mean())}
    res["rescore"] = out
    flagged = [(name_arr[g], int(d), name_arr[ref_train[a]]) for g, d, a in zip(test_idx, dist_train, arg_train) if d <= 4]
    res["examples_within_4_of_train"] = [{"test": a, "distance": b, "train": c} for a, b, c in flagged[:20]]

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump(res, open(args.out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "examples_within_4_of_train"}, indent=1))
