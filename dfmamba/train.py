import argparse
import json
import os
import socket
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .data import FeatureCache
from .diloco import AllReduceEveryStep, DiLoCo
from .metrics import summary
from .model import temporal_from_cfg


@torch.no_grad()
def evaluate(model, path, dev, batch=64):
    ds = FeatureCache(path)
    model.eval()
    ys, ps, loss = [], [], 0.0
    for x, y in DataLoader(ds, batch_size=batch):
        logits = model(x.to(dev)).float()
        loss += F.cross_entropy(logits, y.to(dev), reduction="sum").item()
        ps += logits.softmax(1)[:, 1].tolist()
        ys += y.tolist()
    model.train()
    out = summary(ys, ps)
    out["loss"] = loss / len(ys)
    return out


def batches(loader):
    while True:
        for b in loader:
            yield b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["diloco", "allreduce", "single"], default=None)
    ap.add_argument("--inner-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--transport", choices=["gloo", "hub"], default="gloo")
    ap.add_argument("--hub", default=None, help="HOST:PORT of the hub (transport hub)")
    ap.add_argument("--rank", type=int, default=None)
    ap.add_argument("--world", type=int, default=None)
    ap.add_argument("--hub-timeout", type=float, default=1800.0)
    ap.add_argument("--train-limit", type=int, default=None, help="use at most this many clips from this worker's shard")
    ap.add_argument("--compress", choices=["none", "int8", "int4", "topk"], default="none",
                    help="pseudo-gradient compression on the wire (hub transport only); none = bf16")
    ap.add_argument("--topk-frac", type=float, default=0.1, help="fraction of entries kept by --compress topk")
    ap.add_argument("--no-error-feedback", action="store_true", help="do not carry the compression residual to the next round")
    ap.add_argument("--stochastic", action="store_true", help="stochastic rounding for int8/int4")
    ap.add_argument("--overlap", action="store_true", help="exchange in the background and apply the result one round late (diloco)")
    ap.add_argument("--overlap-delay", type=int, default=None, help="inner steps between sending and applying (default: H)")
    ap.add_argument("--overlap-alpha", type=float, default=0.5, help="weight of the local parameters when merging the late global ones")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    tc = cfg["train"]
    mode = a.mode or tc["mode"]
    H = a.inner_steps or tc["inner_steps"]
    bs = a.batch_size or tc["batch_size"]

    use_hub = a.transport == "hub"
    if use_hub:
        if not a.hub or a.rank is None or a.world is None:
            ap.error("--transport hub needs --hub HOST:PORT, --rank and --world")
        if not 0 <= a.rank < a.world:
            ap.error("--rank must be in 0..world-1")
    distributed = not use_hub and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if distributed:
        dist.init_process_group("gloo")
    if use_hub:
        rank, n = a.rank, a.world
    else:
        rank = dist.get_rank() if distributed else 0
        n = dist.get_world_size() if distributed else 1
    if a.device:
        dev = torch.device(a.device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)) % torch.cuda.device_count())
    else:
        dev = torch.device("cpu")
    torch.manual_seed(tc["seed"])

    data = Path(a.data)
    shard = data / f"train_shard{rank}of{n}.pt"
    if use_hub and shard.exists():
        train = FeatureCache(shard, limit=a.train_limit)
    else:
        train = FeatureCache(data / "train.pt", rank, n, limit=a.train_limit)
    loader = DataLoader(train, batch_size=bs, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(tc["seed"] + rank))
    cfg["model"].setdefault("feat_dim", train.feats.shape[-1])
    model = temporal_from_cfg(cfg).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"], betas=(0.9, 0.95))
    comm_dtype = getattr(torch, tc.get("comm_dtype", "bfloat16"))
    net = {}
    if a.compress != "none" and not use_hub and n > 1:
        ap.error("--compress needs --transport hub")
    if a.overlap and mode != "diloco":
        ap.error("--overlap needs --mode diloco")
    net.update(compress=a.compress, topk_frac=a.topk_frac, error_feedback=not a.no_error_feedback, stochastic=a.stochastic)
    if use_hub:
        net.update(transport="hub", hub=a.hub, rank=rank, world_size=n, hub_timeout=a.hub_timeout,
                   hub_token=os.environ.get("DFM_HUB_TOKEN", ""))
    if mode == "diloco":
        sync = DiLoCo(params, H, tc["outer_lr"], tc["outer_momentum"], comm_dtype, weight=bs,
                      overlap=a.overlap, overlap_delay=a.overlap_delay, overlap_alpha=a.overlap_alpha, **net)
    else:
        sync = AllReduceEveryStep(params, comm_dtype, weight=bs, **net)
        H = tc.get("log_every", 100)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = {"rank": rank, "world": n, "host": socket.gethostname(), "device": str(dev),
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
            "mode": mode, "transport": a.transport, "inner_steps": H, "batch_size": bs, "train_clips_this_worker": len(train),
            "trainable_params": sum(p.numel() for p in params), "payload_MB": sync.payload_bytes() / 1e6,
            "compress": a.compress, "topk_frac": a.topk_frac, "error_feedback": not a.no_error_feedback,
            "overlap": a.overlap}
    print(json.dumps(info), flush=True)
    log = [info]
    it = batches(loader)
    total = tc["total_steps"]
    t_start = time.time()
    run_loss, run_n, t_seg = 0.0, 0, time.time()
    for step in range(1, total + 1):
        x, y = next(it)
        loss = F.cross_entropy(model(x.to(dev)), y.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if mode != "diloco":
            sync.before_step()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sync.step()
        run_loss += loss.item()
        run_n += 1
        if step % H == 0 or step == total:
            rec = {"step": step, "rank": rank, "train_loss": run_loss / run_n, "syncs": sync.syncs,
                   "payload_MB_sent": round(sync.bytes / 1e6, 2), "wire_MB_est": round(sync.wire_bytes() / 1e6, 2),
                   "wait_s": round(sync.wait_seconds, 3), "comm_s": round(sync.comm_seconds, 3),
                   "blocked_s": round(getattr(sync, "blocked_seconds", 0.0), 3),
                   "hidden_s": round(getattr(sync, "hidden_seconds", 0.0), 3),
                   "segment_s": round(time.time() - t_seg, 2), "elapsed_s": round(time.time() - t_start, 2)}
            if rank == 0 and (step // H) % tc.get("eval_every_syncs", 1) == 0 and (data / "val.pt").exists():
                rec["val"] = evaluate(model, data / "val.pt", dev)
            print(json.dumps(rec), flush=True)
            log.append(rec)
            run_loss, run_n, t_seg = 0.0, 0, time.time()
    if sync.pending():
        sync.sync()
        log.append({"final_sync": True, "syncs": sync.syncs, "payload_MB_sent": round(sync.bytes / 1e6, 2),
                    "wire_MB_est": round(sync.wire_bytes() / 1e6, 2), "wait_s": round(sync.wait_seconds, 3),
                    "comm_s": round(sync.comm_seconds, 3), "blocked_s": round(getattr(sync, "blocked_seconds", 0.0), 3),
                    "hidden_s": round(getattr(sync, "hidden_seconds", 0.0), 3), "elapsed_s": round(time.time() - t_start, 2)})
    sync.close()
    (out / f"log_rank{rank}.json").write_text(json.dumps(log, indent=1))
    if rank == 0:
        torch.save({"model": model.state_dict(), "config": cfg}, out / "model.pt")
        if (data / "val.pt").exists():
            (out / "val_metrics.json").write_text(json.dumps(evaluate(model, data / "val.pt", dev), indent=1))
        if (data / "test.pt").exists():
            res = evaluate(model, data / "test.pt", dev)
            (out / "test_metrics.json").write_text(json.dumps(res, indent=1))
            print(json.dumps({"test": res}), flush=True)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
