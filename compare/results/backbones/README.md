# Backbone comparison: EfficientNet-B0 vs ResNet

Setup note. Sections 1 and 2 use a fixed, balanced 40,000-image training subset of MegaDataset (20,000 real + 20,000 fake), a 6,000-image validation subset (3,000 + 3,000) and the FULL test split (10,905 images). The earlier 140k-image results are in results/images_megadataset. Images are 256x256 face crops, resized to 224 for these runs. Section 4 and later use the full dataset (140,002 train / 39,428 validation / 10,905 test) at native 256 px.

## 1. Frozen backbones (ImageNet weights, only a small head is trained)

Head: the no-temporal head from configs/images_no_temporal.yaml (a small MLP, d_model 512), 3000 steps, batch 64. GFLOPs come from torch FlopCounterMode for one 224x224 image (a multiply-add counts as 2 FLOPs). Source: metrics/<backbone>.json, metrics/<backbone>_log_rank0.json, logs/extract_*.log, flops.json.

| Backbone | Feature dim | Backbone params | GFLOPs / image | Head trainable params | Test acc | Test AUC | Best val acc | Extract time, 40k train images |
|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | 1280 | 4.0M | 0.77 | 788,738 | 72.4% | 0.819 | 82.9% | 149s |
| ResNet-18 | 512 | 11.2M | 3.63 | 395,522 | 67.7% | 0.747 | 79.9% | 33s |
| ResNet-50 | 2048 | 23.5M | 8.17 | 1,181,954 | 72.0% | 0.803 | 81.4% | 38s |
| CLIP ViT-L/14 | 768 | 304.0M | 155.5 | not run | not run | not run | not run | not run (skipped to free the GPU for another job; cost columns are measured, accuracy is not) |

Not run: the 4-layer Mamba head on frozen EfficientNet-B0 / ResNet-50 features and the d_model x layers sweep on the 40k-subset frozen features. The jobs were killed when the machine crashed and the temporary feature files were lost. The head-size question is answered instead on the fine-tuned v2 features (section 6).

## 2. Fine-tuned end to end, same recipe (40k subset, 3000 steps, batch 64, lr 3e-4, 224 px, about 4.8 passes over the subset)

Source: metrics/ft_*_log.json and logs/finetune_*.log. Train seconds are wall-clock on a shared RTX 4060 Ti; the EfficientNet-B0 linear figure includes a pause forced by another job, so it is not comparable with the other three.

| Backbone | Head | Total params | Train seconds | Val acc (6k) | Test acc @0.5 | Test AUC | Test acc @ val-chosen threshold |
|---|---|---|---|---|---|---|---|
| EfficientNet-B0 | linear | 4,010,110 | 1756 | 98.2% | 86.8% | 0.958 | not recorded (run predates the flag) |
| EfficientNet-B0 | mamba | 5,737,214 | 324 | 97.7% | 86.2% | 0.957 | 86.4% (thr 0.555) |
| ResNet-50 | linear | 23,512,130 | 465 | 98.0% | 86.2% | 0.942 | 86.0% (thr 0.415) |
| ResNet-50 | mamba | 25,434,306 | 526 | 98.1% | 86.9% | 0.954 | 86.8% (thr 0.485) |

## 3. What the comparison says

Single seed per configuration, so differences under about 1 point are noise.

- Frozen ImageNet features are the weak point. With the same small head, EfficientNet-B0 (72.4%) and ResNet-50 (72.0%) are tied (half a point apart, one seed) and ResNet-18 is clearly worse (67.7%). Test AUC is 0.82, 0.80 and 0.75.
- Fine-tuning the whole network lifts every model to 86-87% test accuracy and 0.94-0.96 AUC, even with only 40k training images. This is the large gain; the backbone choice matters much less than whether it is trained.
- ResNet-50 does not beat EfficientNet-B0. On AUC EfficientNet-B0 is ahead (linear heads 0.958 against 0.942, Mamba heads 0.957 against 0.954). On accuracy at threshold 0.5 they are level: EfficientNet-B0 linear 86.8%, ResNet-50 with a Mamba head 86.9%, ResNet-50 linear 86.2%, EfficientNet-B0 Mamba 86.2%. A 0.1-point gap on 10,905 test images is nothing. EfficientNet-B0 costs 0.77 GFLOPs per 224 image against 8.17 for ResNet-50 (10.6 times less) and has 4.0M parameters against 23.5M. So the README claim "similar accuracy at about 10x less compute" holds in this test. The accuracy-versus-AUC split (ResNet-50 + Mamba matches on accuracy, EfficientNet-B0 is ahead on AUC) is within what one seed can produce.
- On still images the Mamba layers see a sequence of length 1, so they act as extra MLP-like capacity. Their cross-frame role needs video data. The Mamba head did not help EfficientNet-B0 and changed ResNet-50 by +0.7 points of accuracy and +0.012 AUC, which one seed cannot separate from run-to-run noise (no repeated-seed runs were made).
- Validation accuracy is not quoted. Validation overlaps Train by 13.5% (5,334 of 39,428 images are byte-identical to a Train image; `results/dataset_duplicates/dup.json`), so it is used only for checkpoint and threshold selection and no validation accuracy is quoted as a result. Test has no exact copies in Train or Validation.


## 4. v2 improved recipe (full 140k train, 256 px native, strong augmentation, 2 epochs, label smoothing 0.1)

EfficientNet-B0 with a linear head, 4,375 steps of batch 64, lr 3e-4, cosine schedule. Two runs of the identical recipe exist and both are reported:

| Run | Hardware | Train seconds | Val acc (full 39,428) | Test acc @0.5 | Test AUC | Val-chosen threshold | Test acc @ that threshold | Status |
|---|---|---|---|---|---|---|---|---|
| Original local run | RTX 4060 Ti, shared | 7215 | 98.0% | 89.44% | 0.9496 | 0.37 | 88.78% | checkpoint lost in the machine crash; numbers kept from its log |
| 3090 retrain (used for everything downstream) | RTX 3090 | 428 | 98.5% | 89.62% | 0.9583 | 0.42 | 89.30% | checkpoint ~/work/bb_runs/ft_effnet_b0_v2/model.pt |

Said plainly: the original local run reached 89.44% / AUC 0.950 and its checkpoint was lost when WSL crashed and /tmp was wiped. The run was repeated with the exact same recipe on an RTX 3090 and reached 89.62% / AUC 0.958; that retrain is the model used for the DiLoCo rerun, the head sweep and the quantisation test. The two runs differ by 0.2 points of accuracy and 0.009 AUC. They used different hardware, so data order and augmentation draws are not identical; no repeated-seed runs were made, so how much of the gap is run-to-run noise is not measured. Train seconds are not comparable (shared 4060 Ti versus a dedicated 3090).

Validation versus Test: every model scores about 98% on Validation and 86-90% on Test, so the Validation split is much easier than Test. Picking the decision threshold on Validation did not reliably help Test: across the runs that recorded it, Test accuracy moved by between -0.7 and +0.2 points, and it was lower for both v2 runs. The 0.5 threshold is the one to quote.

What changed from the first recipe (section 2) and why:

- Training data: 40k balanced subset -> full 140,002 images, because more data is the cheapest fix for the gap between frozen-feature and fine-tuned results.
- Resolution: 224 -> native 256, so no resize blur on the face crops (the networks accept 256).
- Augmentation (train only): random resized crop (scale 0.8-1.0), horizontal flip, colour jitter, random JPEG re-compression (quality 50-100, p=0.5) and Gaussian blur (p=0.2). JPEG and blur target the Validation-to-Test gap.
- Schedule: 3000 steps -> 2 epochs (4,375 steps) with cosine decay, label smoothing 0.1.
- Thresholds: Test is reported at 0.5 and at the threshold that maximises Validation accuracy; nothing was tuned on Test.
- Data fix: one truncated image (Train/Fake/fake_42153.jpg) from an interrupted copy was replaced from the original (see corrupt_images.json). PIL's LOAD_TRUNCATED_IMAGES is kept on as a safety net. The original v2 run was relaunched after that crash and one image out of 140k cannot change the result.

## 5. DiLoCo rerun on the v2 (3090) fine-tuned features

Features: ~/work/mega_feats_ft (frozen v2 trunk, extracted at 256 px), configs/images.yaml (4 Mamba layers), 4 simulated workers on one GPU. Files: results/images_megadataset_ft/.

| Run (3,000 steps per worker, batch 64) | Inner steps H | Syncs | Payload sent per worker (MB) | Est. ring all-reduce traffic per worker (MB) | Test accuracy | Test AUC |
|---|---|---|---|---|---|---|
| 1 worker, Mamba | - | 0 | 0.0 | 0.0 | 89.2% | 0.936 |
| 4 workers, sync gradients every step | every step | 3000 | 69023.2 | 103534.9 | 88.8% | 0.932 |
| 4 workers, DiLoCo, H = 100 | 100 | 30 | 690.2 | 1035.3 | 87.5% | 0.923 |
| 4 workers, DiLoCo, H = 500 | 500 | 6 | 138.1 | 207.1 | 86.2% | 0.920 |


## 6. Small Mamba heads and int8/int4 on the v2 features

Full tables are in head_sweep_ft.md and head_sweep_ft.json (single run per size, no repeated seeds). Reading it: head size barely matters. From 84k to 1.7M trainable parameters Test accuracy stays between 88.5% and 88.8%; AUC varies between 0.927 and 0.945 without following parameter count (the 64-wide 2-layer head has the best AUC, the 128-wide heads the lowest), which is the size of difference one seed can produce. int8 weights change no prediction on the 10,905 test images and int4 weights flip 5, with accuracy unchanged to within 0.03 points and AUC 0.001 lower, for the d128 L2 head with per-row scales. Only Linear weights were quantised; activations, biases, conv, A_log, D and LayerNorm stayed fp32, so this does not prove a fully integer datapath works.

### Small Mamba heads on v2 fine-tuned EfficientNet-B0 features

Features: ~/work/mega_feats_ft (frozen v2 trunk, 1280-dim, full MegaDataset), configs/images.yaml with the listed d_model / n_layers, single worker, 3000 steps, batch 64. n_layers = 0 is an MLP only. Test = full 10,905-image test split. Input is one frame, so the Mamba layers act as extra MLP-like capacity here.

| d_model | Mamba layers | Trainable params | Test acc | Test AUC | Best val acc |
|---|---|---|---|---|---|
| 256 | 2 | 1,729,666 | 88.8% | 0.936 | 98.6% |
| 128 | 1 | 350,786 | 88.8% | 0.927 | 98.6% |
| 128 | 2 | 528,962 | 88.7% | 0.928 | 98.6% |
| 64 | 1 | 132,386 | 88.7% | 0.933 | 98.6% |
| 64 | 2 | 180,514 | 88.7% | 0.945 | 98.6% |
| 64 | 0 | 84,258 | 88.5% | 0.931 | 98.6% |

#### Hardware target (d128 L2)

Head: d_model 128, 2 Mamba layers, trained on ~/work/mega_feats_ft (run ftsweep_d128_n2). Every nn.Linear weight in the head (516,224 of 528,962 parameters; this includes the input projection, the Mamba in/x/dt/out projections and the classifier layers) is fake-quantised with a symmetric per-output-channel scale (one scale per weight row, group size = full row, round to nearest, no clipping search). Biases, the depthwise conv, A_log, D and LayerNorm stay fp32, and activations stay fp32. Evaluated on the full 10,905-image test split; flips are counted at threshold 0.5 against the fp32 predictions.

| Precision | Test acc | Test AUC | Prediction flips vs fp32 |
|---|---|---|---|
| fp32 | 88.74% | 0.9283 | 0 of 10905 |
| int8 | 88.74% | 0.9283 | 0 of 10905 |
| int4 | 88.77% | 0.9273 | 5 of 10905 |

