# Where this detector fits (checked November 2026)

## Open detectors you can download today

| Detector / benchmark | What it targets | Notes |
|---|---|---|
| [DeepfakeBench](https://github.com/SCLBD/DeepfakeBench) | Face-swap images and video; one protocol across FF++, Celeb-DF, DFDC and others | Re-implements about 35 detectors, including video models (TALL, FTCN, X-CLIP, VideoMAE) and image models (Xception, EfficientNet-B4, UCF, SPSL, SBI, CLIP, Effort). In the original paper, trained on FF++ c23, frame-level cross-dataset AUC averages about 0.70-0.79. CC BY-NC 4.0. |
| [Effort](https://github.com/YZY-stack/Effort-AIGI-Detection) (ICML 2025) | Face deepfakes and AI-generated images | CLIP ViT-L/14 backbone; checkpoints released. CC BY-NC 4.0. |
| [UnivFD](https://github.com/WisconsinAIVision/UniversalFakeDetect) (CVPR 2023) | AI-generated images | Linear probe on CLIP features. MIT. |
| [DeMamba / GenVideo](https://github.com/chenhaoxing/DeMamba) (2024) | Fully generated video (Sora, Pika, SVD, ...) | Also Mamba-based; no pretrained weights in the README. Apache-2.0. |

## How well detectors work outside their training data

- [Deepfake-Eval-2024](https://arxiv.org/abs/2503.02857) (in-the-wild content from 2024): open video detectors score AUC 0.50-0.63 off the shelf. Fine-tuning helps (GenConViT 0.63 -> 0.82). The best commercial video tool reached AUC 0.79.
- [SynthForensics](https://arxiv.org/abs/2602.04939) (2026, videos from 15 open generators): face-forgery detectors lose 13-55 AUC points compared with FF++.

## Commercial tools

Reality Defender and Hive sell detection APIs (image, video, audio), each with a free web tool. Neither publishes accuracy figures. Intel's FakeCatcher (2022) and TrueMedia.org's hosted service (closed January 2025) are not reliably available.

## What that means for this repository

This is a **small, efficient** detector: a frozen EfficientNet-B0 (4.0M parameters, 0.39 billion multiply-adds per frame) plus an 11.5M-parameter Mamba head, trainable on consumer GPUs with DiLoCo. It is not expected to beat CLIP-ViT-L-based detectors such as Effort on generalization. The fair comparisons are:

1. **Same protocol:** the DeepfakeBench FF++ c23 split, with cross-dataset tests on Celeb-DF and DFDC, against its CNN baselines (Xception, EfficientNet-B4).
2. **Cost:** parameters, milliseconds per clip (see [inference-kernels](https://github.com/chanakyagt/inference-kernels)), and training traffic.
3. **At least one out-of-distribution test.**
