import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--world", type=int, required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    d = torch.load(a.src, map_location="cpu")
    out = Path(a.out) if a.out else Path(a.src).parent
    out.mkdir(parents=True, exist_ok=True)
    for r in range(a.world):
        idx = torch.arange(len(d["labels"]))[r::a.world]
        path = out / f"train_shard{r}of{a.world}.pt"
        torch.save({"feats": d["feats"][idx].clone(), "labels": d["labels"][idx].clone()}, path)
        print(f"{path}  {len(idx)} clips  {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
