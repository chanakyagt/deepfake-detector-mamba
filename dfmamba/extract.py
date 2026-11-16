import argparse
import time
from pathlib import Path

import torch

from .data import FaceCropper, clip_tensor, list_videos, sample_frames
from .model import FrameEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--frames", type=int, default=30)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = FrameEncoder(pretrained=True).to(dev)
    if dev.type == "cuda":
        enc = enc.to(memory_format=torch.channels_last)
    crop = FaceCropper()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for split in a.splits.split(","):
        vids = list_videos(a.videos, split)
        feats, labels, names, skipped, nfaces = [], [], [], [], []
        t0 = time.time()
        for k, (path, label) in enumerate(vids):
            faces = [f for f in (crop.crop(fr) for fr in sample_frames(path, a.frames)) if f is not None]
            x = clip_tensor(faces, a.frames)
            if x is None:
                skipped.append(path.name)
                continue
            with torch.no_grad(), torch.autocast(dev.type, torch.bfloat16, enabled=dev.type == "cuda"):
                f = enc(x[None].to(dev))
            feats.append(f[0].half().cpu())
            labels.append(label)
            names.append(f"{path.parent.name}/{path.name}")
            nfaces.append(min(len(faces), a.frames))
            if (k + 1) % 50 == 0:
                print(f"{split}: {k + 1}/{len(vids)} clips, {time.time() - t0:.0f}s", flush=True)
        if not feats:
            continue
        torch.save({"feats": torch.stack(feats), "labels": torch.tensor(labels), "names": names,
                    "frames": a.frames, "skipped": skipped, "faces_found": torch.tensor(nfaces)}, Path(a.out) / f"{split}.pt")
        nf, lab = torch.tensor(nfaces).float(), torch.tensor(labels)
        print(f"{split}: saved {len(feats)} clips, skipped {len(skipped)} with no face; mean faces found "
              f"real {nf[lab == 0].mean():.1f}, fake {nf[lab == 1].mean():.1f} of {a.frames}", flush=True)


if __name__ == "__main__":
    main()
