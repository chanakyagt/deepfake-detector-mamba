import argparse
from pathlib import Path

import torch


def clips(n, frames, dim, latent, swaps, gen, W):
    z = torch.zeros(n, frames, latent)
    z[:, 0] = torch.randn(n, latent, generator=gen)
    for t in range(1, frames):
        z[:, t] = 0.9 * z[:, t - 1] + 0.436 * torch.randn(n, latent, generator=gen)
    labels = torch.zeros(n, dtype=torch.long)
    labels[n // 2:] = 1
    for i in range(n // 2, n):
        for _ in range(swaps):
            a, b = torch.randint(0, frames, (2,), generator=gen).tolist()
            z[i, [a, b]] = z[i, [b, a]]
    feats = torch.relu(z @ W + 0.1 * torch.randn(n, frames, dim, generator=gen))
    perm = torch.randperm(n, generator=gen)
    return feats[perm].half(), labels[perm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=4000)
    ap.add_argument("--val", type=int, default=800)
    ap.add_argument("--test", type=int, default=800)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--swaps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    gen = torch.Generator().manual_seed(a.seed)
    W = torch.randn(64, 1280, generator=gen) / 8
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for split, n in (("train", a.train), ("val", a.val), ("test", a.test)):
        f, y = clips(n, a.frames, 1280, 64, a.swaps, gen, W)
        torch.save({"feats": f, "labels": y, "names": [f"synthetic_{split}_{i}" for i in range(n)],
                    "frames": a.frames, "skipped": [], "synthetic": True}, Path(a.out) / f"{split}.pt")
        print(split, tuple(f.shape), int(y.sum()), "fake")


if __name__ == "__main__":
    main()
