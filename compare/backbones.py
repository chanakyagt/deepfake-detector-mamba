import torch
import torch.nn as nn
from torchvision import models

IMAGENET = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
CLIP = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])


class TorchvisionTrunk(nn.Module):
    def __init__(self, name, pretrained=True):
        super().__init__()
        if name == "effnet_b0":
            net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self.body, self.dim = nn.Sequential(net.features, net.avgpool, nn.Flatten(1)), 1280
        elif name == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            net.fc = nn.Identity()
            self.body, self.dim = net, 512
        elif name == "resnet50":
            net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            net.fc = nn.Identity()
            self.body, self.dim = net, 2048
        else:
            raise ValueError(name)

    def forward(self, x):
        return self.body(x)


class ClipTrunk(nn.Module):
    def __init__(self, path):
        super().__init__()
        from transformers import CLIPVisionModelWithProjection
        self.body = CLIPVisionModelWithProjection.from_pretrained(path, torch_dtype=torch.float32)
        self.dim = self.body.config.projection_dim

    def forward(self, x):
        return self.body(pixel_values=x).image_embeds


def build(name, clip_path=None, pretrained=True):
    if name == "clip_vitl14":
        return ClipTrunk(clip_path), 224, CLIP
    return TorchvisionTrunk(name, pretrained), 224, IMAGENET


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def count_flops(m, x):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        m(x)
    return fc.get_total_flops()
