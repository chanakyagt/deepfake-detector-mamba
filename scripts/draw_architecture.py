import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

boxes = [
    ("Video", "30 frames\nsampled uniformly", "#eef3fb"),
    ("Face crop", "largest face, 25% margin\n224 x 224 RGB", "#eef3fb"),
    ("EfficientNet-B0", "ImageNet, frozen\n1 frame -> 1280", "#eef3fb"),
    ("Feature cache", "30 x 1280 fp16\n77 KB per clip", "#fff4e0"),
    ("Linear", "1280 -> 512", "#fdeeee"),
    ("Mamba x4", "selective scan\nstate 1024 x 16", "#fdeeee"),
    ("Mean + MLP", "512 -> 256 -> 2", "#fdeeee"),
]
fig, ax = plt.subplots(figsize=(13, 2.9), dpi=150)
w, gap = 1.55, 0.35
for i, (title, sub, fc) in enumerate(boxes):
    x = i * (w + gap)
    ax.add_patch(FancyBboxPatch((x, 0), w, 1.2, boxstyle="round,pad=0.04", fc=fc, ec="#1f4e99"))
    ax.text(x + w / 2, 0.82, title, ha="center", va="center", fontsize=9, weight="bold")
    ax.text(x + w / 2, 0.38, sub, ha="center", va="center", fontsize=7)
    if i:
        ax.add_patch(FancyArrowPatch((x - gap, 0.6), (x, 0.6), arrowstyle="->", mutation_scale=10))
ax.text(1 * (w + gap) + w / 2 + 0.95, -0.3, "run once per clip: dfmamba/extract.py", ha="center", fontsize=7, color="#555")
ax.text(5 * (w + gap) + w / 2, -0.3, "trained with DiLoCo across GPUs: dfmamba/train.py", ha="center", fontsize=7, color="#555")
ax.set_xlim(-0.2, len(boxes) * (w + gap)); ax.set_ylim(-0.5, 1.35); ax.axis("off")
fig.tight_layout()
fig.savefig("docs/architecture.png")
