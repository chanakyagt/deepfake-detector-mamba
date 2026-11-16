import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from .ssm import MambaLayer

FEAT_DIM = 1280


class FrameEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        net = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = net.features
        self.pool = net.avgpool
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def forward(self, frames):
        b, t = frames.shape[:2]
        f = self.pool(self.features(frames.flatten(0, 1))).flatten(1)
        return f.view(b, t, -1)


class TemporalClassifier(nn.Module):
    def __init__(self, d_model=512, n_layers=4, d_state=16, dropout=0.1, num_classes=2, temporal=True, feat_dim=FEAT_DIM):
        super().__init__()
        self.proj = nn.Linear(feat_dim, d_model)
        self.layers = nn.ModuleList([MambaLayer(d_model, d_state) for _ in range(n_layers if temporal else 0)])
        self.norm_f = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(d_model // 2, num_classes))

    def forward(self, feats):
        h = self.proj(feats)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm_f(h).mean(1))


class Detector(nn.Module):
    def __init__(self, pretrained=True, **kw):
        super().__init__()
        self.encoder = FrameEncoder(pretrained)
        self.temporal = TemporalClassifier(**kw)

    def forward(self, frames):
        return self.temporal(self.encoder(frames))


def temporal_from_cfg(cfg):
    m = cfg["model"]
    return TemporalClassifier(m["d_model"], m["n_layers"], m["d_state"], m["dropout"], 2, m.get("temporal", True),
                              m.get("feat_dim", FEAT_DIM))
