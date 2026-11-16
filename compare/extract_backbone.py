import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare.backbones import build
from dfmamba.extract_images import ImageFolderFaces


class RawImages(Dataset):
    def __init__(self, items, size):
        self.items, self.size = items, size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, label = self.items[i]
        img = Image.open(p).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1), label, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["effnet_b0", "resnet18", "resnet50", "clip_vitl14"])
    ap.add_argument("--clip-path", default=None)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="Train:train,Validation:val,Test:test")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda")
    net, size, (mean, std) = build(a.backbone, a.clip_path)
    if a.size:
        size = a.size
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cpu")["model"]
        net.load_state_dict({k[len("trunk."):]: v for k, v in sd.items() if k.startswith("trunk.")})
    net = net.to(dev).eval()
    mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for pair in a.splits.split(","):
        src, dst = pair.split(":")
        items = ImageFolderFaces(a.images, src).items
        ds = RawImages(items, size)
        feats = torch.empty(len(ds), 1, net.dim, dtype=torch.float16)
        t0 = time.time()
        for x, _, idx in DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True):
            x = (x.to(dev, non_blocking=True).float() / 255.0 - mean) / std
            with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
                f = net(x)
            feats[idx, 0] = f.float().half().cpu()
        labels = torch.tensor([l for _, l in items])
        torch.save({"feats": feats, "labels": labels, "frames": 1, "backbone": a.backbone,
                    "source": str(a.images), "split": src}, Path(a.out) / f"{dst}.pt")
        print(f"{a.backbone} {src}: {len(ds)} images, dim {net.dim}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
