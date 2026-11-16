import numpy as np


def auc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    npos, nneg = int(y.sum()), int((1 - y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    ranks = np.empty(len(p))
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and ps[j + 1] == ps[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def summary(y, p, thr=0.5):
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"n": len(y), "accuracy": (tp + tn) / max(len(y), 1), "auc": auc(y, p),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}
