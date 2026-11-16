import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class FaceCropper:
    def __init__(self, yunet=None):
        self.mp = self.yn = self.haar = None
        yunet = yunet or os.environ.get("DFM_YUNET")
        if yunet and Path(yunet).exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yn = cv2.FaceDetectorYN.create(str(yunet), "", (320, 320), 0.7)
                self.method = "yunet"
                return
            except cv2.error:
                self.yn = None
        try:
            import mediapipe as mp
            self.mp = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            self.method = "mediapipe"
            return
        except ImportError:
            pass
        if hasattr(cv2, "CascadeClassifier"):
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.method = "haar"
            return
        raise RuntimeError("no face detector: set DFM_YUNET to a YuNet .onnx file, or install mediapipe or OpenCV 4.x")

    def boxes(self, bgr):
        ih, iw = bgr.shape[:2]
        if self.yn is not None:
            self.yn.setInputSize((iw, ih))
            _, faces = self.yn.detect(bgr)
            return [] if faces is None else [tuple(int(v) for v in f[:4]) for f in faces]
        if self.mp is not None:
            res = self.mp.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            out = []
            for d in res.detections or []:
                r = d.location_data.relative_bounding_box
                out.append((int(r.xmin * iw), int(r.ymin * ih), int(r.width * iw), int(r.height * ih)))
            return out
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return [tuple(int(v) for v in b) for b in self.haar.detectMultiScale(g, 1.1, 5, minSize=(40, 40))]

    def crop(self, bgr, size=224, margin=0.25):
        bs = self.boxes(bgr)
        if not bs:
            return None
        x, y, w, h = max(bs, key=lambda b: b[2] * b[3])
        cx, cy, s = x + w / 2, y + h / 2, max(w, h) * (1 + margin)
        H, W = bgr.shape[:2]
        x0, y0 = int(max(cx - s / 2, 0)), int(max(cy - s / 2, 0))
        x1, y1 = int(min(cx + s / 2, W)), int(min(cy + s / 2, H))
        if x1 <= x0 or y1 <= y0:
            return None
        face = cv2.resize(bgr[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)


def sample_frames(path, n):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idx = np.linspace(0, max(total - 1, 0), n).round().astype(int) if total > 0 else []
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return frames


def clip_tensor(faces, n):
    if not faces:
        return None
    faces = (faces + [faces[-1]] * n)[:n]
    a = (np.stack(faces).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(a).permute(0, 3, 1, 2)


def list_videos(root, split):
    items = []
    for label, cls in ((0, "real"), (1, "fake")):
        items += [(p, label) for p in sorted((Path(root) / split / cls).glob("*.mp4"))]
    return items


class FeatureCache(Dataset):
    def __init__(self, path, rank=0, world=1, limit=None):
        d = torch.load(path, map_location="cpu")
        idx = torch.arange(len(d["labels"]))[rank::world][:limit]
        self.feats = d["feats"][idx]
        self.labels = d["labels"][idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.feats[i].float(), self.labels[i]
