# Zero-shot evaluation of pretrained deepfake detectors

Validation note: Validation figures in this document are used only for selecting edits, thresholds or combiner weights and are not headline results. Exact-copy overlap between Validation and Train (13.5%, 5,334 of 39,428 images) was measured on the full MegaDataset Validation split (`results/dataset_duplicates/dup.json`); the 6,000-image Validation subset used here was not checked separately. Quote Test numbers.

No training or fine-tuning. Every model is a downloaded checkpoint run as-is on the MegaDataset face crops (256x256) with `compare/sota_zero_shot.py`.

Test = full Test split, 10,905 images (5,413 Real, 5,492 Fake). Val = the full Validation split of `~/data/MegaSub`, 6,000 images (3,000 per class); the planned 4,000-image subset was not used, the full 6,000 were. Images read from `~/data/MegaSub` (a fixed random subset whose Test folder is the full Test split).

| Model | Trained on | Params (M) | Input | Test acc @0.5 | Test AUC | Val acc @0.5 | Val AUC | Mean p_fake real / fake (Test) | Missing / unexpected keys |
|---|---|---|---|---|---|---|---|---|---|
| Xception (DeepfakeBench, 2023) | FF++ c23 | 21.86 | 256x256 | 0.5075 | 0.4714 | 0.5238 | 0.5605 | 0.389 / 0.376 | 0 / 0 |
| EfficientNet-B4 (DeepfakeBench, 2023) | FF++ c23 | 17.55 | 256x256 | 0.5033 | 0.4980 | 0.5170 | 0.4953 | 0.354 / 0.360 | 0 / 0 |
| CNNDetection/CNNAug ResNet-34 (Wang 2020, via DeepfakeBench) | FF++ c23 | 21.29 | 256x256 | 0.4856 | 0.4840 | 0.4810 | 0.4527 | 0.748 / 0.733 | 0 / 0 |
| SPSL (CVPR 2021) | FF++ c23 | 21.86 | 256x256 | 0.5245 | 0.5325 | 0.4942 | 0.5338 | 0.534 / 0.561 | 0 / 0 |
| F3-Net (ECCV 2020) | FF++ c23 | 22.52 | 256x256 | 0.4690 | 0.4296 | 0.4973 | 0.4680 | 0.449 / 0.399 | 0 / 0 |
| UCF (ICCV 2023) | FF++ c23 | 46.84 | 256x256 | 0.5062 | 0.4494 | 0.5253 | 0.5205 | 0.298 / 0.277 | 0 / 0 |
| UnivFD (CVPR 2023) | ProGAN | 303.97 | 224x224 center-crop | 0.5465 | 0.6271 | 0.5958 | 0.7725 | 0.166 / 0.270 | 0 / 0 |
| Effort (ICML 2025) | FF++ | 504.61 | 224x224 | 0.6013 | 0.6575 | 0.6960 | 0.7657 | 0.468 / 0.605 | 288 / 0 |

All numbers come from the JSON files in this folder (`<model>_Test.json`, `<model>_Validation.json`). Per-image probabilities for the ensemble work are in `probs/<model>_<split>.npz` (`p` float32 probability of fake, `y` label, `names` relative paths, same order as `ImageFolderFaces`).

## Reading these results

- These are cross-dataset tests. Six models were trained on FaceForensics++ (c23) and UnivFD on ProGAN-generated images; none has seen MegaDataset. Different generators, compression and face cropping mean the learned cues often do not transfer.
- Accuracy at the 0.5 threshold is mostly near 50% (chance) because the score threshold is miscalibrated for this data: for most models the mean p_fake is about the same on real and fake, so 0.5 splits the images almost arbitrarily. AUC is threshold-free and shows whether the ranking carries any signal.
- The six DeepfakeBench FF++ detectors (Xception, EfficientNet-B4, ResNet-34, SPSL, F3-Net, UCF) have AUC between 0.43 and 0.53 on Test, i.e. no usable signal (below 0.5 means the ranking is slightly inverted). Only UnivFD (Test AUC 0.627, Val 0.773) and Effort (Test 0.658, Val 0.766) rank real vs fake above chance, and both use a frozen or LoRA/SVD-adapted CLIP ViT-L/14 backbone.
- Validation AUC is clearly higher than Test AUC for UnivFD and Effort (and for Xception), so the two splits are not equally hard; do not read Val as an unbiased proxy for Test.

## Checkpoint loading checks

- Missing and unexpected keys are 0 for all DeepfakeBench detectors and UnivFD (the UnivFD CLIP text-tower weights are not loaded by design since only the vision tower is used; the linear head loads strictly).
- Effort reports 288 missing keys: the `U_r`, `S_r`, `V_r` buffers of `SVDResidualLinear` (24 layers x q/k/v/out projections x 3). They are frozen copies of principal SVD components used only by the training-time orthogonality loss; the forward pass uses `weight_main` plus the residual terms and the bias, all of which are in the checkpoint (confirmed from the Effort source; the JSON lists only a 5-key sample, all `_r` entries). Unexpected keys are 0.
- Effort's parameter count (504.6M) includes the unused CLIP text tower; UnivFD's 303.97M is the vision tower plus projection and head.

## Preprocessing check

- DeepfakeBench detectors: RGB, resolution 256 from the config, mean 0.5 / std 0.5, matching DeepfakeBench's own `abstract_dataset.py` (RGB via cv2 BGR->RGB, cubic resize, `T.Normalize`). The dataset is already 256x256 so no resize happens. Effort uses its config resolution (224) with cubic resize from 256.
- UnivFD: center crop 224, CLIP mean/std, matching `validate.py` (`CenterCrop(224)` + `Normalize`).
- The loader does not apply face-alignment or landmark cropping; the dataset provides pre-cropped faces.

## Latency (fp32, batch 1, RTX 4060 Ti shared with other jobs, so indicative only)

Taken from `latency_ms_b1_fp32` and `throughput_img_s_b64_fp32` in each JSON.

- xception: 3.72 ms, 441.8 img/s
- efficientnetb4: 13.17 ms, 232.7 img/s
- cnnaug_resnet34: 2.78 ms, 1050.1 img/s
- spsl: 3.31 ms, 432.8 img/s
- f3net: 4.18 ms, 416.8 img/s
- ucf: 7.22 ms, 220.3 img/s
- univfd: 21.98 ms, 55.5 img/s
- effort: 24.76 ms, 54.5 img/s
