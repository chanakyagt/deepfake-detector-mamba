import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare.backbones import build, count_params
from compare.extract_backbone import RawImages
from dfmamba.extract_images import ImageFolderFaces
from dfmamba.metrics import summary
from dfmamba.model import TemporalClassifier


class AugImages(RawImages):
    def __init__(self, items, size):
        super().__init__(items, size)
        from torchvision import transforms as T
        self.tf = T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.9, 1.1111), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.RandomApply([T.GaussianBlur(5, (0.1, 1.5))], p=0.2),
        ])

    def __getitem__(self, i):
        import io
        import random
        from PIL import Image
        p, label = self.items[i]
        img = Image.open(p).convert("RGB")
        img = self.tf(img)
        if random.random() < 0.5:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=random.randint(50, 100))
            buf.seek(0)
            img = Image.open(buf).convert("RGB")
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1), label, i


class Classifier(nn.Module):
    def __init__(self, trunk, head="linear"):
        super().__init__()
        self.trunk = trunk
        self.kind = head
        if head == "mamba":
            self.head = TemporalClassifier(d_model=256, n_layers=2, d_state=16, feat_dim=trunk.dim)
        else:
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(trunk.dim, 2))

    def forward(self, x):
        f = self.trunk(x)
        if self.kind == "mamba":
            f = f.unsqueeze(1)
        return self.head(f)


@torch.no_grad()
def evaluate(model, items, size, norm, dev, batch=256, workers=4, return_probs=False):
    model.eval()
    ys, ps = [], []
    for x, y, _ in DataLoader(RawImages(items, size), batch_size=batch, num_workers=workers, pin_memory=True):
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(norm(x.to(dev, non_blocking=True)))
        ps += logits.float().softmax(1)[:, 1].tolist()
        ys += y.tolist()
    model.train()
    if return_probs:
        return summary(ys, ps), ps
    return summary(ys, ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["effnet_b0", "resnet18", "resnet50"])
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", choices=["linear", "mamba"], default="linear")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--size", type=int, default=None)
    ap.add_argument("--aug", choices=["none", "strong"], default="none")
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-limit", type=int, default=6000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda")
    trunk, size, (mean, std) = build(a.backbone)
    if a.size:
        size = a.size
    model = Classifier(trunk, a.head).to(dev).to(memory_format=torch.channels_last)
    m = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(std, device=dev).view(1, 3, 1, 1)

    def norm(x):
        return ((x.float() / 255.0 - m) / s).contiguous(memory_format=torch.channels_last)

    train_items = ImageFolderFaces(a.images, "Train").items
    val_items = ImageFolderFaces(a.images, "Validation", limit=a.val_limit).items
    test_items = ImageFolderFaces(a.images, "Test").items
    if a.epochs:
        a.steps = int(a.epochs * len(train_items) // a.batch)
    train_cls = AugImages if a.aug == "strong" else RawImages
    loader = DataLoader(train_cls(train_items, size), batch_size=a.batch, shuffle=True, drop_last=True,
                        num_workers=a.workers, pin_memory=True, persistent_workers=True,
                        generator=torch.Generator().manual_seed(a.seed))
    opt = torch.optim.AdamW([{"params": model.trunk.parameters(), "lr": a.lr},
                             {"params": model.head.parameters(), "lr": a.lr * 10}], weight_decay=0.05)
    base = [g["lr"] for g in opt.param_groups]
    warm = max(a.steps // 20, 1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = [{"backbone": a.backbone, "head": a.head, "params": count_params(model), "steps": a.steps, "batch": a.batch, "size": size, "aug": a.aug, "label_smoothing": a.label_smoothing, "epochs": a.epochs,
            "train_images_seen": a.steps * a.batch, "train_images_available": len(train_items),
            "gpu": torch.cuda.get_device_name(dev)}]
    print(json.dumps(log[0]), flush=True)
    step, t0, run = 0, time.time(), 0.0
    while step < a.steps:
        for x, y, _ in loader:
            step += 1
            f = step / warm if step <= warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (a.steps - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            x = norm(x.to(dev, non_blocking=True))
            if a.aug == "none":
                flip = torch.rand(x.shape[0], device=dev) < 0.5
                x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
            with torch.autocast("cuda", torch.bfloat16):
                loss = F.cross_entropy(model(x).float(), y.to(dev), label_smoothing=a.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run += loss.item()
            if step % 250 == 0 or step == a.steps:
                rec = {"step": step, "train_loss": round(run / (250 if step % 250 == 0 else step % 250), 4),
                       "elapsed_s": round(time.time() - t0, 1)}
                if step % 1000 == 0 or step == a.steps:
                    rec["val"] = evaluate(model, val_items, size, norm, dev)
                print(json.dumps(rec), flush=True)
                log.append(rec)
                run = 0.0
            if step >= a.steps:
                break
    train_s = time.time() - t0
    test, test_p = evaluate(model, test_items, size, norm, dev, workers=a.workers, return_probs=True)
    val_full, val_p = evaluate(model, ImageFolderFaces(a.images, "Validation").items, size, norm, dev, workers=a.workers, return_probs=True)
    root = Path(a.images)
    np.save(out / "val_p_fake.npy", np.array(val_p, dtype=np.float32))
    np.save(out / "test_p_fake.npy", np.array(test_p, dtype=np.float32))
    (out / "val_names.json").write_text(json.dumps([str(p.relative_to(root)) for p, _ in ImageFolderFaces(a.images, "Validation").items]))
    (out / "test_names.json").write_text(json.dumps([str(p.relative_to(root)) for p, _ in test_items]))
    vy = [l for _, l in ImageFolderFaces(a.images, "Validation").items]
    ty = [l for _, l in test_items]
    vp_a, tp_a = np.array(val_p), np.array(test_p)
    best_thr, best_acc = 0.5, -1.0
    for thr in np.arange(0.02, 0.981, 0.005):
        acc = float(((vp_a > thr).astype(int) == np.array(vy)).mean())
        if acc > best_acc:
            best_thr, best_acc = float(thr), acc
    test_tuned = summary(ty, tp_a, best_thr)
    test_tuned["threshold_from_validation"] = round(best_thr, 3)
    test_tuned["val_accuracy_at_threshold"] = best_acc
    (out / "test_metrics_tuned_threshold.json").write_text(json.dumps(test_tuned, indent=1))
    log.append({"train_seconds": round(train_s, 1), "test": test, "test_tuned_threshold": test_tuned, "val_full": val_full})
    print(json.dumps(log[-1]), flush=True)
    (out / "log.json").write_text(json.dumps(log, indent=1))
    (out / "test_metrics.json").write_text(json.dumps(test, indent=1))
    torch.save({"model": model.state_dict(), "backbone": a.backbone, "head": a.head}, out / "model.pt")


if __name__ == "__main__":
    main()
