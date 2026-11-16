import os
CK = os.environ.get("DFM_CKPT_ROOT", "/tmp")
SC = os.environ.get("DFM_SCRATCH", "/tmp")
import torch
p = torch.load(os.path.join(SC, "wsp_t1_pca.pt"))
mu, V, ev = p["mu"], p["V"], p["ev"]
C = (V * ev) @ V.T
a = torch.load(os.path.join(SC, "wsp_t1_prac.pt"))
b = torch.load(os.path.join(SC, "wsp_t1_cov.pt"))
X = torch.cat([o["inputs"][:, 0] for o in a] + [x for x, _, _ in b])
y = torch.cat([o["labels"] for o in a] + [l for _, l, _ in b])
sd = torch.load(os.path.join(CK, "img_runs", "diloco_H100/model.pt"), map_location="cpu", weights_only=False)["model"]
s1 = torch.load(os.path.join(CK, "img_runs", "single_H1/model.pt"), map_location="cpu", weights_only=False)["model"]
W0 = torch.load(os.path.join(SC, "wsp_init_seed0.pt"))["proj.weight"]
mhat = mu / mu.norm()
print("|mu|", float(mu.norm()), "std along mean dir", float((mhat @ C @ mhat).sqrt()), "median per-PC std", float(ev.sqrt().median()))
for nm, W in (("diloco", sd["proj.weight"]), ("single", s1["proj.weight"]), ("init", W0), ("dW_diloco", sd["proj.weight"] - W0)):
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    print(nm, "top singular values", [round(float(s), 2) for s in S[:6]])
    for i in range(4):
        v = Vh[i]
        var = float(v @ C @ v)
        proj = X @ v
        gap = float((proj[y == 1].mean() - proj[y == 0].mean()) / proj.std())
        print(f"   sv{i}: sigma={float(S[i]):.2f} cos(v,mean_dir)={float(v @ mhat):+.3f} feature std along v={var**0.5:.4f} "
              f"(PC-rank equiv: {int((ev.sqrt() > var**0.5).sum())}) mean x.v={float(proj.mean()):+.3f} "
              f"std-response sigma*std={float(S[i])*var**0.5:.3f} class gap (fake-real)/std={gap:+.3f}")
W = sd["proj.weight"]
U, S, Vh = torch.linalg.svd(W, full_matrices=False)
v1 = Vh[0]
print("input dims with largest |v1| loadings", torch.topk(v1.abs(), 8))
print("feature mean and std on those dims", [(round(float(mu[j]), 3), round(float(X[:, j].std()), 3)) for j in torch.topk(v1.abs(), 8).indices])
print("sign pattern of v1: frac positive", float((v1 > 0).float().mean()), "sum", float(v1.sum()), "cos with all-ones", float(v1.sum() / (1280 ** 0.5)))
print("u1 = output direction of sv1; |u1 . proj.bias|", float(U[:, 0] @ sd["proj.bias"]))
torch.save({"U": U[:, :8], "S": S[:8], "Vh": Vh[:8]}, os.path.join(SC, "wsp_t1_svd.pt"))
