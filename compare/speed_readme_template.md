# Size, dimensionality and speed of every detector

Measured on one RTX 4060 Ti (8 GB), torch 2.14 cu130, by `compare/speed.py` (raw numbers: `speed.json`, per-model files in `raw/`). Rows are sorted by GFLOPs per image.

## Measurement caveat (read first)

The GPU was shared with other jobs. The final timing pass took `/tmp/gpu_heavy.lock` and waited for GPU utilisation of 5% or less on three samples in a row, which happened at 10:38 UTC. The whole pass took about 5 minutes. Utilisation was back at 100% by the end of the pass, and the Windows `nvidia-smi` process list (37 desktop and app processes, which never empties) is stored in each record. So later models in the pass (the downloaded detectors and Effort) may have run partly while the GPU was shared. Treat latencies as indicative to roughly +/-30%, and treat the GFLOPs, parameter and dimension columns as exact.

## How to read the table

- "Ours" rows are named `ours_d<head width>_L<mamba layers>_T<frames>`. `T1` is a single image. `T30` is a 30-frame clip. For `T30`, the b=1 latency is the time for the whole clip (one clip = 30 frames). The b=64 columns are frames per second: a batch of 64 single images for `T1`, and 2 clips (60 frames) for `T30`. The b=64 peak memory column is that batch.
- GFLOPs are per frame or image, counted with `torch.utils.flop_counter.FlopCounterMode` (a multiply-add counts as 2 FLOPs). It counts only matrix multiplies and convolutions. The Mamba scan (Triton kernel, or the Python reference loop) is plain element-wise maths and is not counted. It is tiny next to the convolutions, but the "ours" GFLOPs are slightly under-counted.
- Peak memory is `torch.cuda.max_memory_allocated` and includes the model weights.
- Latency: median of 50 timed runs after 10 warm-up runs, with `cuda.synchronize`. Throughput: median of 20 runs after 5 warm-up runs. bf16 means `torch.autocast` to bfloat16. fp32 uses the PyTorch default (no TF32 for matmuls).
- Params: "Trainable" for our rows is the head only (the EfficientNet-B0 trunk, 4.01M parameters, is frozen and shared by every "ours" row). For `frozen_*` rows it is a stand-in head for timing, not the real trained head. For the downloaded detectors it is every parameter that has `requires_grad` in the code as loaded. Effort counts both its frozen original weights and its SVD copies, so its total (504.6M) is larger than the 303M parameters in the checkpoint file. Its trainable part is 0.2M.
- Effort loaded with 288 missing keys. These are the SVD-residual tensors that are rebuilt at load time, which does not affect speed. I did not check them against accuracy.
- The "SSM state" column is the Mamba recurrent state: layers x (2 x head width) x 16 floats, in fp32 KB. A convolution cache of 3 values per channel per layer (12,288 floats for the default head) comes on top. A CNN has no recurrent state, so it shows n/a.

## What "dimensionality" means here

Follow one image through the model:

1. **Input pixels**: 224 x 224 x 3 = 150,528 numbers (the CNN-Aug, Xception, EfficientNet-B4, SPSL, F3Net and UCF detectors read 256 x 256 x 3 = 196,608).
2. **Feature vector**: the backbone squeezes the picture into one list of numbers. Ours gives 1280 per frame. ResNet-18 gives 512, ResNet-50 gives 2048, CLIP ViT-L/14 gives 768, and the downloaded detectors give 512 to 2048.
3. **Head width**: our head first projects 1280 down to 512 (or 256, 128, 64). A smaller width means fewer parameters and a smaller state.
4. **State size**: each Mamba layer carries a memory of (2 x width) x 16 numbers from frame to frame. The default head has 4 layers at width 512, which is 65,536 numbers (256 KB in fp32). This memory is fixed. A 30-frame clip does not grow it.

So 150,528 pixels become 1280 features, then 512, then 65,536 state numbers (spread over 4 layers). The final answer is 2 numbers (real or fake).

## Table

{{TABLE}}

`univfd` and `frozen_clip_vitl14` are the same CLIP ViT-L/14 backbone. `univfd` adds a single Linear(768,1) probe. `effort` is the same ViT-L/14 with SVD-split attention weights. Their GFLOPs are within 0.1% of each other.

The bar chart is not produced because matplotlib is not installed in the venv and nothing may be downloaded.
