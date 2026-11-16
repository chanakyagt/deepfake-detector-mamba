import os
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HOME = Path.home()
SOTA = HOME / "sota"
CLIP_PATH = str(SOTA / "hf_clip" / "clip-vit-large-patch14")
DFB_TRAIN = str(SOTA / "DeepfakeBench" / "training")
EFFORT_TRAIN = str(SOTA / "Effort" / "DeepfakeBench" / "training")
DFB_W = str(SOTA / "dfb_weights")
PYDEPS = str(SOTA / "pydeps")
EFFORT_CKPT = SOTA / "effort_weights" / "effort_clip_L14_trainOn_FaceForensic.pth"
FC = str(SOTA / "UnivFD" / "pretrained_weights" / "fc_weights.pth")

OURS_CONFIGS = [("ours_d512_L4", 512, 4, True), ("ours_d256_L2", 256, 2, True), ("ours_d128_L2", 128, 2, True),
                ("ours_d64_L0", 64, 0, True), ("ours_d512_notemporal", 512, 4, False)]
DFB_NAMES = ["xception", "efficientnetb4", "cnnaug_resnet34", "spsl", "f3net", "ucf"]


def nparams(m, trainable=False):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))


def sync():
    torch.cuda.synchronize()


def median_ms(fn, iters, warm):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        sync()
        ts = []
        for _ in range(iters):
            t = time.perf_counter()
            fn()
            sync()
            ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000


def gflops(fn):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        fn()
    return fc.get_total_flops() / 1e9


def feature_hook(model):
    calls = []

    def hook(mod, inp, out):
        calls.append(inp[0].shape[-1])

    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Linear)]
    return calls, hs


def bench(make_input, call, units_per_item, big_batch, dev):
    res = {}
    x1 = make_input(1)
    res["gflops_per_unit"] = round(gflops(lambda: call(x1)) / units_per_item, 3)
    for tag, dtype in (("fp32", None), ("bf16", torch.bfloat16)):
        def run(x):
            if dtype is None:
                return call(x)
            with torch.autocast("cuda", dtype=dtype):
                return call(x)
        for b in (1, big_batch):
            key = f"{tag}_b{b}"
            try:
                torch.cuda.empty_cache()
                x = make_input(b)
                torch.cuda.reset_peak_memory_stats()
                if b == 1:
                    ms = median_ms(lambda: run(x), 50, 10)
                    res[f"latency_ms_{key}"] = round(ms, 3)
                else:
                    ms = median_ms(lambda: run(x), 20, 5)
                    res[f"throughput_units_per_s_{key}"] = round(b * units_per_item / ms * 1000, 1)
                res[f"peak_mem_MB_{key}"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
                del x
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                res[f"oom_{key}"] = True
    return res


def base_rec(name, model, size, extra):
    return {"model": name, "params_total": nparams(model), "params_trainable": nparams(model, True),
            "input_resolution": size, **extra}


def run_ours(only=None):
    from dfmamba.model import Detector
    dev = torch.device("cuda")
    out = []
    for name, d, layers, temporal in OURS_CONFIGS:
        det = Detector(pretrained=False, d_model=d, n_layers=layers, temporal=temporal).to(dev).eval()
        calls, hs = feature_hook(det)
        with torch.no_grad():
            det(torch.randn(1, 2, 3, 224, 224, device=dev))
        classifier_in = calls[-1]
        for h in hs:
            h.remove()
        n_layers = len(det.temporal.layers)
        d_inner = 2 * d
        state_floats = n_layers * d_inner * 16
        conv_floats = n_layers * d_inner * 3
        for T, big in ((1, 64), (30, 2)):
            rec = base_rec(f"{name}_T{T}", det, "224x224 (T frames)", {
                "frames_per_input": T, "feature_dim_from_backbone": 1280, "head_width": d,
                "classifier_input_dim": classifier_in, "mamba_layers": n_layers,
                "ssm_state_floats": state_floats, "ssm_state_KB_fp32": round(state_floats * 4 / 1024, 1),
                "conv_state_floats": conv_floats, "big_batch": big, "unit": "frame",
                "params_backbone_frozen": nparams(det.encoder), "params_head_trainable": nparams(det.temporal)})
            rec.update(bench(lambda b: torch.randn(b, T, 3, 224, 224, device=dev), lambda x: det(x), T, big, dev))
            out.append(rec)
        del det
        torch.cuda.empty_cache()
    return out


def run_backbones(names):
    from compare.backbones import build
    dev = torch.device("cuda")
    out = []
    for name in names:
        trunk, size, _ = build(name, CLIP_PATH, pretrained=False)
        trunk = trunk.to(dev).eval()
        head = nn.Sequential(nn.Linear(trunk.dim, 512), nn.LayerNorm(512), nn.Linear(512, 256), nn.GELU(),
                             nn.Linear(256, 2)).to(dev).eval()
        net = nn.Sequential(trunk, head)
        for p in trunk.parameters():
            p.requires_grad = False
        rec = base_rec(f"frozen_{name}", net, f"{size}x{size}", {
            "frames_per_input": 1, "feature_dim_from_backbone": trunk.dim, "head_width": 512,
            "classifier_input_dim": trunk.dim, "ssm_state_floats": 0, "big_batch": 64, "unit": "frame",
            "params_backbone_frozen": nparams(trunk), "params_head_trainable": nparams(head),
            "note": "head here is a stand-in Linear(dim,512)-LN-Linear(512,256)-GELU-Linear(256,2) for timing; see compare/finetune.py for the real one"})
        rec.update(bench(lambda b: torch.randn(b, 3, size, size, device=dev), lambda x: net(x), 1, 64, dev))
        out.append(rec)
        del net, trunk, head
        torch.cuda.empty_cache()
    return out


def run_downloaded(name):
    from compare import sota_zero_shot as sz
    dev = torch.device("cuda")
    scratch = os.environ.get("DFM_SCRATCH", "/tmp")
    if name in sz.DFB:
        model, cfg, res = sz.build_dfb(name, DFB_TRAIN, DFB_W, scratch, PYDEPS)
        size = cfg["resolution"]
        label = lambda x: torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        model = model.to(dev).eval()
        call = lambda x: model({"image": x, "label": label(x)}, inference=True)["cls"]
        extra = {"missing_keys": len(res.missing_keys), "unexpected_keys": len(res.unexpected_keys)}
        resolution = f"{size}x{size}"
    elif name == "effort":
        from transformers import CLIPModel, CLIPVisionModel
        model, cfg, res = sz.build_effort(EFFORT_TRAIN, str(EFFORT_CKPT), CLIP_PATH, PYDEPS)
        size = cfg["resolution"]
        model = model.to(dev).eval()
        call = lambda x: model({"image": x}, inference=True)["cls"]
        extra = {"missing_keys": len(res.missing_keys), "unexpected_keys": len(res.unexpected_keys)}
        resolution = f"{size}x{size}"
    elif name == "univfd":
        model = sz.UnivFD(CLIP_PATH, FC).to(dev).eval()
        size = 224
        call = lambda x: model(x)
        extra = {"note": "CLIP ViT-L/14 frozen + one Linear(768,1) probe"}
        resolution = "224x224 center-crop"
    else:
        raise ValueError(name)
    calls, hs = feature_hook(model)
    with torch.no_grad():
        call(torch.randn(2, 3, size, size, device=dev))
    for h in hs:
        h.remove()
    rec = base_rec(name, model, resolution, {
        "frames_per_input": 1, "classifier_input_dim": calls[-1] if calls else None,
        "linear_input_dims_seen_last5": calls[-5:], "ssm_state_floats": 0, "big_batch": 64, "unit": "image",
        **extra})
    rec.update(bench(lambda b: torch.randn(b, 3, size, size, device=dev), call, 1, 64, dev))
    return [rec]


def gpu_state():
    q = lambda args: subprocess.run(args, capture_output=True, text=True).stdout.strip()
    return {"util_mem": q(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader"]),
            "windows_process_count": len(q(["/mnt/c/Windows/System32/nvidia-smi.exe", "--query-compute-apps=pid",
                                            "--format=csv,noheader"]).splitlines()),
            "windows_process_list": q(["/mnt/c/Windows/System32/nvidia-smi.exe",
                                       "--query-compute-apps=pid,process_name", "--format=csv,noheader"]).splitlines()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shared", default="unknown")
    a = ap.parse_args()
    if a.target == "ours":
        recs = run_ours()
    elif a.target in ("resnet18", "resnet50", "clip_vitl14"):
        recs = run_backbones([a.target])
    else:
        recs = run_downloaded(a.target)
    state = gpu_state()
    for r in recs:
        r["gpu"] = torch.cuda.get_device_name(0)
        r["measured_while_gpu_shared"] = a.shared
        r["gpu_state_after"] = state
        r["torch"] = torch.__version__
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(recs, indent=1))
    print(json.dumps([{k: r.get(k) for k in ("model", "params_total", "gflops_per_unit", "latency_ms_bf16_b1")}
                      for r in recs]), flush=True)


if __name__ == "__main__":
    main()
