import os
import argparse
import importlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfmamba.extract_images import ImageFolderFaces
from dfmamba.metrics import summary

DFB = {
    "xception": ("xception", "xception_best.pth"),
    "efficientnetb4": ("efficientnetb4", "effnb4_best.pth"),
    "cnnaug_resnet34": ("resnet34", "cnnaug_best.pth"),
    "spsl": ("spsl", "spsl_best.pth"),
    "f3net": ("f3net", "f3net_best.pth"),
    "ucf": ("ucf", "ucf_best.pth"),
}


class Images(Dataset):
    def __init__(self, items, size, crop):
        self.items, self.size, self.crop = items, size, crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, label = self.items[i]
        img = Image.open(p).convert("RGB")
        if self.crop:
            w, h = img.size
            l, t = (w - self.size) // 2, (h - self.size) // 2
            img = img.crop((l, t, l + self.size, t + self.size))
        elif img.size != (self.size, self.size):
            img = img.resize((self.size, self.size), Image.BICUBIC)
        return torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1), label


class AnyLoss:
    def __getitem__(self, k):
        return lambda *a, **kw: nn.CrossEntropyLoss()


def install_stubs(training_dir, extra_paths):
    for p in extra_paths:
        sys.path.insert(0, p)
    sys.path.insert(0, training_dir)
    sk = types.ModuleType("sklearn")
    sk.metrics = types.ModuleType("sklearn.metrics")
    sys.modules["sklearn"], sys.modules["sklearn.metrics"] = sk, sk.metrics
    tb = types.ModuleType("torch.utils.tensorboard")
    tb.SummaryWriter = object
    sys.modules["torch.utils.tensorboard"] = tb
    from metrics.registry import DETECTOR
    det = types.ModuleType("detectors")
    det.__path__ = [str(Path(training_dir) / "detectors")]
    det.DETECTOR = DETECTOR
    sys.modules["detectors"] = det
    loss = types.ModuleType("loss")
    loss.LOSSFUNC = AnyLoss()
    sys.modules["loss"] = loss
    orig = torchvision.models.resnet34
    torchvision.models.resnet34 = lambda *a, **k: orig(weights=None)
    return DETECTOR


def clean_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    return {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}


def build_dfb(name, training_dir, weights_dir, scratch, pydeps=None):
    DETECTOR = install_stubs(training_dir, [pydeps] if pydeps else [])
    cfg_name, ckpt = DFB[name]
    cfg = yaml.safe_load(open(Path(training_dir) / "config" / "detector" / f"{cfg_name}.yaml"))
    dummy = Path(scratch) / "dummy_pretrained.pth"
    torch.save({"conv1.weight": torch.zeros(32, 3, 3, 3)}, dummy)
    cfg["pretrained"] = "" if cfg_name == "efficientnetb4" else str(dummy)
    mod_file = {"xception": "xception_detector", "efficientnetb4": "efficientnetb4_detector",
                "resnet34": "resnet34_detector", "spsl": "spsl_detector", "f3net": "f3net_detector",
                "ucf": "ucf_detector"}[cfg_name]
    importlib.import_module(f"detectors.{mod_file}")
    model = DETECTOR[cfg["model_name"]](cfg)
    res = model.load_state_dict(clean_state(Path(weights_dir) / ckpt), strict=False)
    return model, cfg, res


def build_effort(training_dir, ckpt, clip_path, pydeps):
    from transformers import CLIPModel
    DETECTOR = install_stubs(training_dir, [pydeps])
    mod = importlib.import_module("detectors.effort_detector")
    orig = mod.CLIPModel

    class Clip:
        @staticmethod
        def from_pretrained(_p, **kw):
            return orig.from_pretrained(clip_path, **kw)

    mod.CLIPModel = Clip
    cfg = yaml.safe_load(open(Path(training_dir) / "config" / "detector" / "effort.yaml"))
    model = DETECTOR["effort"](cfg)
    res = model.load_state_dict(clean_state(ckpt), strict=False)
    return model, cfg, res


class UnivFD(nn.Module):
    def __init__(self, clip_path, fc_path):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection
        self.clip = CLIPVisionModelWithProjection.from_pretrained(clip_path, torch_dtype=torch.float32)
        self.fc = nn.Linear(self.clip.config.projection_dim, 1)
        self.fc.load_state_dict(torch.load(fc_path, map_location="cpu", weights_only=True))

    def forward(self, x):
        return self.fc(self.clip(pixel_values=x).image_embeds)


def time_model(fn, x, iters=50):
    with torch.no_grad():
        for _ in range(10):
            fn(x)
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            fn(x)
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t)
    return float(np.median(ts) * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--split", default="Test")
    ap.add_argument("--training-dir")
    ap.add_argument("--weights")
    ap.add_argument("--clip-path")
    ap.add_argument("--pydeps")
    ap.add_argument("--scratch", default=os.environ.get("DFM_SCRATCH", "/tmp"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--probs-out")
    a = ap.parse_args()
    dev = torch.device("cuda")
    if a.model in DFB:
        model, cfg, res = build_dfb(a.model, a.training_dir, a.weights, a.scratch, a.pydeps)
        size, crop = cfg["resolution"], False
        mean, std = cfg["mean"], cfg["std"]
        prob = lambda x: model({"image": x, "label": torch.zeros(x.shape[0], dtype=torch.long, device=x.device)},
                               inference=True)["cls"].float().softmax(1)[:, 1]
        source = f"DeepfakeBench v1.0.1 release, {DFB[a.model][1]}, trained on FF++ c23"
    elif a.model == "effort":
        model, cfg, res = build_effort(a.training_dir, a.weights, a.clip_path, a.pydeps)
        size, crop = cfg["resolution"], False
        mean, std = cfg["mean"], cfg["std"]
        prob = lambda x: model({"image": x}, inference=True)["cls"].float().softmax(1)[:, 1]
        source = "Effort (ICML 2025), effort_clip_L14_trainOn_FaceForensic.pth, trained on FF++"
    elif a.model == "univfd":
        model = UnivFD(a.clip_path, a.weights)
        res = None
        size, crop = 224, True
        from compare.backbones import CLIP
        mean, std = CLIP
        prob = lambda x: model(x).float().sigmoid()[:, 0]
        source = "UnivFD (CVPR 2023), fc_weights.pth, trained on ProGAN images"
    else:
        raise ValueError(a.model)
    model = model.to(dev).eval()
    m = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    s = torch.tensor(std, device=dev).view(1, 3, 1, 1)
    norm = lambda x: (x.float() / 255.0 - m) / s
    items = ImageFolderFaces(a.images, a.split, limit=a.limit).items
    ys, ps = [], []
    t0 = time.time()
    with torch.no_grad():
        for x, y in DataLoader(Images(items, size, crop), batch_size=a.batch, num_workers=a.workers, pin_memory=True):
            ps += prob(norm(x.to(dev, non_blocking=True))).cpu().tolist()
            ys += y.tolist()
    eval_s = time.time() - t0
    out = summary(ys, ps)
    if a.probs_out:
        Path(a.probs_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.probs_out, p=np.asarray(ps, dtype=np.float32), y=np.asarray(ys, dtype=np.int64),
                 names=np.asarray([str(Path(p).relative_to(a.images)) for p, _ in items]))
    params = sum(p.numel() for p in model.parameters())
    x1 = torch.randn(1, 3, size, size, device=dev)
    x64 = torch.randn(64, 3, size, size, device=dev)
    rec = {"model": a.model, "source": source, "split": a.split, "images": len(ys),
           "input": f"{size}x{size}{' center-crop' if crop else ''}", "params_M": round(params / 1e6, 2),
           "missing_keys": len(res.missing_keys) if res else 0,
           "unexpected_keys": len(res.unexpected_keys) if res else 0,
           "metrics": out, "mean_p_fake_on_real": float(np.mean([p for p, y in zip(ps, ys) if y == 0])),
           "mean_p_fake_on_fake": float(np.mean([p for p, y in zip(ps, ys) if y == 1])),
           "latency_ms_b1_fp32": round(time_model(prob, x1), 2),
           "throughput_img_s_b64_fp32": round(64 / time_model(prob, x64, 20) * 1000, 1),
           "eval_seconds": round(eval_s, 1), "gpu": torch.cuda.get_device_name(dev)}
    if res is not None and (res.missing_keys or res.unexpected_keys):
        rec["missing_sample"] = res.missing_keys[:5]
        rec["unexpected_sample"] = res.unexpected_keys[:5]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
