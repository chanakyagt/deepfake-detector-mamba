import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dfmamba.data import FaceCropper, clip_tensor, sample_frames
from dfmamba.model import FrameEncoder, temporal_from_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=30)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(a.ckpt, map_location="cpu")
    head = temporal_from_cfg(ck["config"]).to(dev).eval()
    head.load_state_dict(ck["model"])
    enc = FrameEncoder(pretrained=True).to(dev)
    crop = FaceCropper()
    out = []
    for v in a.videos:
        faces = [f for f in (crop.crop(fr) for fr in sample_frames(v, a.frames)) if f is not None]
        if not faces:
            out.append({"video": Path(v).name, "folder": Path(v).parent.name, "faces": 0})
            print(json.dumps(out[-1]), flush=True)
            continue
        x = clip_tensor(faces, len(faces))
        with torch.no_grad():
            feats = enc(x[None].to(dev))
            p = head(feats[0][:, None]).softmax(1)[:, 1].cpu()
        out.append({"video": Path(v).name, "folder": Path(v).parent.name, "faces": len(faces),
                    "p_fake_mean": round(p.mean().item(), 3), "p_fake_min": round(p.min().item(), 3),
                    "p_fake_max": round(p.max().item(), 3), "frames_over_0.5": int((p > 0.5).sum())})
        print(json.dumps(out[-1]), flush=True)


if __name__ == "__main__":
    main()
