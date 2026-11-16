import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .data import MEAN, STD
from .model import FrameEncoder

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_dir(root, name):
    for d in Path(root).iterdir():
        if d.is_dir() and d.name.lower() == name.lower():
            return d
    return None


class ImageFolderFaces(Dataset):
    def __init__(self, root, split, limit=None, seed=0):
        base = find_dir(root, split)
        items = []
        for label, cls in ((0, "real"), (1, "fake")):
            d = find_dir(base, cls) if base else None
            files = sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS) if d else []
            if limit:
                rng = np.random.default_rng(seed + label)
                files = [files[i] for i in sorted(rng.choice(len(files), min(limit // 2, len(files)), replace=False))]
            items += [(p, label) for p in files]
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, label = self.items[i]
        img = Image.open(p).convert("RGB").resize((224, 224), Image.BILINEAR)
        a = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
        return torch.from_numpy(a).permute(2, 0, 1), label, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="Train:train,Validation:val,Test:test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = FrameEncoder(pretrained=True).to(dev).to(memory_format=torch.channels_last)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for pair in a.splits.split(","):
        src, dst = pair.split(":")
        ds = ImageFolderFaces(a.images, src, a.limit if dst == "train" else None)
        if not len(ds):
            print(f"{src}: no images found")
            continue
        feats = torch.empty(len(ds), 1, 1280, dtype=torch.float16)
        t0 = time.time()
        for x, _, idx in DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True):
            with torch.no_grad(), torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
                f = enc(x.to(dev, non_blocking=True).contiguous(memory_format=torch.channels_last)[:, None])
            feats[idx] = f.half().cpu()
        labels = torch.tensor([l for _, l in ds.items])
        torch.save({"feats": feats, "labels": labels, "names": [str(p.relative_to(a.images)) for p, _ in ds.items],
                    "frames": 1, "source": str(a.images), "split": src}, Path(a.out) / f"{dst}.pt")
        print(f"{src} -> {dst}.pt: {len(ds)} images ({int(labels.sum())} fake) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
