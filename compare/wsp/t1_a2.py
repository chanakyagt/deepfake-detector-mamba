import os
CK = os.environ.get("DFM_CKPT_ROOT", "/tmp")
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import torch
a = torch.load(os.path.join(SC, "wsp_t1_prac.pt"))
b = torch.load(os.path.join(SC, "wsp_t1_cov.pt"))
X = torch.cat([o["inputs"][:, 0] for o in a] + [x for x, _, _ in b])
y = torch.cat([o["labels"] for o in a] + [l for _, l, _ in b])
L = torch.cat([o["logits"] for o in a] + [g for _, _, g in b])
ld = L[:, 1] - L[:, 0]
print("N", len(y), "fake", int(y.sum()))
for c in (0, 1):
    q = ld[y == c]
    print("class", c, "logit-diff mean", round(float(q.mean()), 3), "std", round(float(q.std()), 3), "quantiles 5/25/50/75/95",
          [round(float(v), 2) for v in torch.quantile(q, torch.tensor([.05, .25, .5, .75, .95]))])
print("practice acc", float(((ld > 0).long() == y).float().mean()))
mu = X.mean(0)
Xc = X - mu
C = Xc.T @ Xc / (len(X) - 1)
ev, V = torch.linalg.eigh(C)
ev, V = ev.flip(0).clamp(min=0), V.flip(1)
cumv = ev.cumsum(0) / ev.sum()
for q in (0.9, 0.95, 0.99, 0.999):
    print("PCs for", q, ":", int((cumv < q).sum()) + 1)
k = int((cumv < 0.99).sum()) + 1
sd = torch.load(os.path.join(CK, "img_runs", "diloco_H100/model.pt"), map_location="cpu", weights_only=False)["model"]
W = sd["proj.weight"]
P = V[:, :k] @ V[:, :k].T
Wl = W - W @ P
print("k", k, "|W|", float(W.norm()), "|W P|", float((W @ P).norm()), "|W (I-P)|", float(Wl.norm()))
print("energy fraction of proj.weight outside the 99% subspace", float(Wl.norm() ** 2 / W.norm() ** 2))
print("on practice data: |W xc| total", float((Xc @ W.T).norm()), " part from outside subspace", float((Xc @ Wl.T).norm()))
g_in = (W @ V[:, :k]).norm(dim=0) * ev[:k].sqrt()
g_out = (W @ V[:, k:]).norm(dim=0) * ev[k:].sqrt()
print("per-direction response std: inside median", float(g_in.median()), "outside median", float(g_out.median()))
print("gain (|W v|) inside median", float((W @ V[:, :k]).norm(dim=0).median()), "outside median", float((W @ V[:, k:]).norm(dim=0).median()))
print("worst-case: a unit-norm shift along the outside subspace moves h0 by up to", float(torch.linalg.svdvals(W @ V[:, k:])[0]),
      "; along inside top PC", float((W @ V[:, 0]).norm()))
print("input-feature total std (sqrt trace C)", float(ev.sum().sqrt()), "outside-subspace std", float(ev[k:].sum().sqrt()))
torch.save({"mu": mu, "V": V, "ev": ev, "k": k, "n": len(X)}, os.path.join(SC, "wsp_t1_pca.pt"))
