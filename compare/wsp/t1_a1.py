import os
CK = os.environ.get("DFM_CKPT_ROOT", "/tmp")
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import torch
outs = torch.load(os.path.join(SC, "wsp_t1_prac.pt"))
cat = lambda k: torch.cat([o["acts"][k] for o in outs])
y = torch.cat([o["labels"] for o in outs])
X = torch.cat([o["inputs"] for o in outs])[:, 0]
L = torch.cat([o["logits"] for o in outs])
ld = L[:, 1] - L[:, 0]
print("N", len(y), "fake frac", float(y.float().mean()))
print("input feat: mean norm", float(X.norm(dim=1).mean()), "per-dim std median", float(X.std(0).median()), "min", float(X.std(0).min()), "max", float(X.std(0).max()))
print("input feat norm real vs fake", float(X[y == 0].norm(dim=1).mean()), float(X[y == 1].norm(dim=1).mean()))
sd = torch.load(os.path.join(CK, "img_runs", "diloco_H100/model.pt"), map_location="cpu", weights_only=False)["model"]
r = cat("proj")[:, 0]
print("residual norms: proj", float(r.norm(dim=1).mean()))
for i in range(4):
    m = cat(f"layers.{i}.mixer")[:, 0]
    s = cat(f"layers.{i}")[:, 0]
    print(f"layer {i}: |mixer out| {float(m.norm(dim=1).mean()):.2f}  |stream| {float(s.norm(dim=1).mean()):.2f}  cos(mixer,stream_in) {float(torch.nn.functional.cosine_similarity(m, s - m, 1).mean()):.3f}")
h1 = cat("head.1")[:, 0] if cat("head.1").dim() == 3 else cat("head.1")
w = sd["head.3.weight"]; b = sd["head.3.bias"]
wd = w[1] - w[0]
contrib = h1 * wd
print("logit diff reconstruct err", float((contrib.sum(1) + b[1] - b[0] - ld).abs().max()))
sep = contrib[y == 1].mean(0) - contrib[y == 0].mean(0)
order = sep.abs().argsort(descending=True)
print("hidden units sorted by class-separation of their logit contribution; top 10:")
tot = sep.abs().sum()
cum = 0
for j in order[:10]:
    cum += sep[j].abs()
    print(int(j), f"sep={float(sep[j]):.3f}  cum%={100*float(cum/tot):.1f}  w={float(wd[j]):.3f}  meanact={float(h1[:, j].mean()):.3f}  sd={float(h1[:, j].std()):.3f}")
for k in (5, 10, 20, 50, 100):
    print("top", k, "units carry", round(100 * float(sep.abs()[order[:k]].sum() / tot), 1), "% of class separation")
nf = cat("norm_f")[:, 0]
print("norm_f out: per-channel |mean| top", nf.mean(0).abs().topk(5))
print("norm_f channel std top", nf.std(0).topk(5))
mu_r, mu_f = X[y == 0].mean(0), X[y == 1].mean(0)
dmu = mu_f - mu_r
sdp = X.std(0)
dz = dmu / (sdp + 1e-6)
print("input dims with largest standardized class gap", dz.abs().topk(10))
C = torch.cov(X.T)
ev, V = torch.linalg.eigh(C)
ev, V = ev.flip(0), V.flip(1)
cumv = ev.cumsum(0) / ev.sum()
for q in (0.5, 0.8, 0.9, 0.95, 0.99):
    print("PCs for", q, "variance:", int((cumv < q).sum()) + 1)
W = sd["proj.weight"]
gain = (W @ V).norm(dim=0)
print("proj gain on PCs 1,2,5,10,50,100,200,400,500:", [round(float(gain[i]), 3) for i in (0, 1, 4, 9, 49, 99, 199, 399, 499)])
print("sqrt eig on same:", [round(float(ev[i].clamp(min=0).sqrt()), 4) for i in (0, 1, 4, 9, 49, 99, 199, 399, 499)])
W0 = torch.load(os.path.join(SC, "wsp_init_seed0.pt"))["proj.weight"]
g0 = (W0 @ V).norm(dim=0)
print("init proj gain same PCs:", [round(float(g0[i]), 3) for i in (0, 1, 4, 9, 49, 99, 199, 399, 499)])
print("frac of proj.weight energy in top-100 / top-300 PCs:", float((W @ V[:, :100]).norm() ** 2 / W.norm() ** 2), float((W @ V[:, :300]).norm() ** 2 / W.norm() ** 2))
dW = W - W0
print("frac of dW energy in top-100 / top-300 / top-511 PCs:", float((dW @ V[:, :100]).norm() ** 2 / dW.norm() ** 2), float((dW @ V[:, :300]).norm() ** 2 / dW.norm() ** 2), float((dW @ V[:, :511]).norm() ** 2 / dW.norm() ** 2))
lp = (dW @ V[:, :511]) @ V[:, :511].T
print("fraction of practice h0 variance explained... |dW x| vs |dW_lowvar x|", float((X @ dW.T).norm()), float((X @ (dW - lp).T).norm()))
print("ld real", float(ld[y==0].mean()), float(ld[y==0].std()), "fake", float(ld[y==1].mean()), float(ld[y==1].std()))
print("names sample", outs[0]["names"][:2], outs[0]["names"][-2:])
torch.save({"V": V, "ev": ev, "mu": X.mean(0), "sd": sdp}, os.path.join(SC, "wsp_t1_pca.pt"))
