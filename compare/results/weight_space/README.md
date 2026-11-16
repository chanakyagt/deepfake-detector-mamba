# Weight-space programming: setting a pretrained detector's weights directly

Validation note: Validation figures in this document are used only for selecting edits, thresholds or combiner weights and are not headline results. Exact-copy overlap between Validation and Train (13.5%, 5,334 of 39,428 images) was measured on the full MegaDataset Validation split (`results/dataset_duplicates/dup.json`); the 6,000-image Validation subset used here was not checked separately. Quote Test numbers.

STATUS: STOPPED EARLY (final). The GPU was reprioritised for higher-priority runs and the remaining jobs were cancelled, so this is the complete set of what ran. Finished: dimension probe (all 5 models); k=0 and the label-only edits (bias, ridge, centroid, LDA, WiSE-style interpolation) for Xception, EfficientNet-B4 and CNN-Aug ResNet-34; the weights-written-equals-closed-form check for those three; and AdaBN with N=1000 unlabelled Train images (+ all head edits on top of it) for CNN-Aug ResNet-34 only. Nothing below is projected or invented; missing cells were not run.

## What was NOT run (GPU reprioritised, jobs cancelled)

- UnivFD and Effort: probe and Test/Validation feature caches exist, and ridge/LDA hyperparameters were tuned for UnivFD (`tune_univfd.json`), but no edit results (k=0 or k>0) and no weights-written check were produced.
- AdaBN with N=1000 unlabelled images: not run for Xception and EfficientNet-B4.
- AdaBN-k (BN statistics from the k labelled images, plus AdaBN-k + head): not run for any model.
- WiSE-style interpolation on top of AdaBN: never planned.
- k=5 was not run (k=4 and k=8 bracket it).
Reason for the gaps: each full-Test AdaBN pass took about 5 minutes on a GPU shared by four jobs, and the shared lock queue starved these jobs for over an hour at a time.

## What was done, in plain language

Idea tested: take a detector trained on a different dataset (FaceForensics++), learn its layer sizes by pushing 5 images through it, then change its numbers directly (no training loop, no gradients) so that it works on MegaDataset.

Edits are made on the final layer (the "head") or on BatchNorm statistics. The trunk (everything before the head) is never changed.

- Head size: the last layer maps a feature vector of D numbers to the real/fake decision. The two-class DeepfakeBench heads store 2xD weights + 2 biases; only the difference (one weight vector w of D numbers and one bias b) matters. Numbers written per edit: Xception 4,098; EfficientNet-B4 3,586; CNN-Aug ResNet-34 1,026; UnivFD 769; Effort 2,050. Of these, D+1 are free (2,049 / 1,793 / 513 / 769 / 1,025); the other half is the mirrored copy (W0 = -w/2, W1 = +w/2).
- AdaBN rewrites BatchNorm running mean and variance: Xception 41 BN layers, EfficientNet-B4 96, ResNet-34 36 (counts in `probe_*.json`; element counts from `Model.edit_counts()`).
- Edit methods (all in `compare/weight_space.py`):
  - (a) bias recalibration: only the final bias is shifted so that the k labelled images are split at the midpoint of the two class mean scores (`bias`), or at the accuracy-maximising threshold on those k images (`bias_opt`).
  - (b) AdaBN, label-free: BN statistics are recomputed on k target images, or on N=1000 unlabelled Train images (`adabn1000`).
  - (c) closed-form head, written into the final Linear: ridge regression to +-1 targets (`ridge`), nearest class centroid (`centroid`), shrinkage LDA (`lda`).
  - (d) AdaBN + closed-form head (`adabn_k/*`, `adabn1000/*`).
  - (e) WiSE-FT-style linear interpolation between the original head and the ridge or LDA head, alpha 0.25/0.5/0.75 (`wise_ridge_a`, `wise_lda_a`). Raw weight interpolation, no rescaling.
- Verification that "weights written" equals "closed-form classifier": for seed 0, k=32, every head was written into the real network's Linear layer and 256 Test images were pushed through the full network. The network's score (logit1 - logit0) was compared with features @ w + b. Max absolute difference: Xception 0.031, EfficientNet-B4 0.010, ResNet-34 0.009 (float32 network vs float64 reference; the smallest per-head maximum score magnitudes are 2.13, 0.55 and 1.03, so the worst-case relative deviation is about 2%); sign agreement 100% in every case (`verify_*.json`).
- Hyperparameters (ridge lambda, LDA shrinkage) were chosen on a 2,000-image Validation subset (original trunk, k=8/32/128, 10 draws), never on Test (`tune_*.json`). Selected: Xception ridge 0.001, LDA 0.01; EfficientNet-B4 ridge 0.1, LDA 0.9. See `tune_*.json` for the others.

## Protocol

- k images = k/2 real + k/2 fake, drawn from Train (class-balanced, nested across k within a seed). k in {4, 8, 32, 128, 1024}; 5 seeds (0-4). k=5 was not run; k=4 and k=8 bracket it.
- Evaluation: the full Test split, 10,905 images, 256x256 crops. Accuracy uses score > 0 (probability 0.5); AUC uses the raw score. Std is the sample std over the 5 seeds.
- Train images were sampled from the full Train folder at /mnt/c/.../DatasetConverter/MegaDataset (not from ~/data/MegaSub, which was not ready when sampling happened). Test is the same 10,905-image split.
- Preprocessing: DeepfakeBench models at 256 px with mean/std 0.5; UnivFD 224 center crop with CLIP stats; Effort 256 to 224 bicubic (antialiased torch resize, close to but not identical to PIL) with CLIP stats. Test trunk features were cached once per model; head methods then need only k forward passes.
- Reference: our own trained model reached 71-73% Test accuracy (AUC about 0.78-0.82) after full training on 140k images with frozen EfficientNet-B0 features.

## Dimension probe (5 images)

5 Test images were pushed through each model with forward hooks on every module. Full per-layer tables: `probe_*.json` (all modules) and `probe_tables.md` (top-level blocks plus head). Summary:

| model | input | penultimate feature | head W, b | BN layers | parameters |
|---|---|---|---|---|---|
| Xception (DFB) | 5x3x256x256 | 5x2048 (after ReLU + global avg pool) | [2,2048], [2] | 41 | 21,861,162 |
| EfficientNet-B4 (DFB) | 5x3x256x256 | 5x1792 | [2,1792], [2] | 96 | 17,552,202 |
| CNN-Aug ResNet-34 (DFB) | 5x3x256x256 | 5x512 | [2,512], [2] | 36 | 21,285,698 |
| UnivFD (CLIP ViT-L/14) | 5x3x224x224 | 5x768 (image_embeds) | [1,768], [1] | 0 | 303,966,977 |
| Effort (CLIP ViT-L/14 + SVD residual) | 5x3x224x224 | 5x1024 (pooler_output) | [2,1024], [2] | 0 | 303,378,530 |

All DeepfakeBench checkpoints loaded with 0 missing and 0 unexpected keys. Effort: 0 missing keys. Note the probe only gives shapes; it does not tell the head anything about real vs fake. All label information comes from the k labelled images.

## Results (3 DeepfakeBench CNNs, label-only edits)

Test accuracy / AUC of the original networks (k=0, 10,905 Test images): Xception 0.507 / 0.471; EfficientNet-B4 0.503 / 0.498; CNN-Aug ResNet-34 0.486 / 0.484. These are chance level on this dataset; Xception and ResNet-34 AUC below 0.5 means their ordering is slightly inverted.

Best mean over the edit methods that finished (full grid with std in `tables.md`, raw rows in `rows_*.json`, aggregates in `summary.json`):

| model | k | best accuracy | best AUC |
|---|---|---|---|
| Xception | 4 | centroid 0.506+-0.035 | LDA 0.513+-0.026 |
| Xception | 1024 | ridge 0.581+-0.011 | ridge 0.615+-0.010 |
| EfficientNet-B4 | 4 | WiSE ridge a=0.25 0.503+-0.005 | bias 0.498+-0.000 (unchanged from original) |
| EfficientNet-B4 | 1024 | ridge 0.545+-0.001 | ridge 0.572+-0.002 |
| CNN-Aug ResNet-34 | 4 | centroid 0.501+-0.010 | centroid 0.492+-0.014 |
| CNN-Aug ResNet-34 | 1024 | AdaBN-1000 + LDA 0.590+-0.005 (without AdaBN: LDA 0.581+-0.007) | AdaBN-1000 + LDA 0.627+-0.008 (without AdaBN: LDA 0.613+-0.009) |

AdaBN (CNN-Aug ResNet-34 only, BN statistics recomputed from 1,000 unlabelled Train images, 5 seeds): AdaBN alone moves the original head from acc 0.486 / AUC 0.484 to 0.481 / 0.453, so recomputing BN statistics made the unedited detector slightly worse on this data. AdaBN + closed-form head at k=1024: LDA 0.590+-0.005 acc, 0.627+-0.008 AUC; ridge 0.586+-0.009 / 0.619+-0.011. That is +0.009 accuracy and +0.014 AUC over the same head on the original trunk (LDA 0.581 / 0.613), about one to two seed standard deviations, from one model. At k=4 AdaBN + head is 0.47-0.49 accuracy, no better than chance. Rows are the `adabn1000/` prefix in `tables.md`.

Intermediate k (accuracy, ridge head): Xception 0.503 / 0.518 / 0.529 / 0.546 / 0.581 for k = 4 / 8 / 32 / 128 / 1024; EfficientNet-B4 0.491 / 0.493 / 0.521 / 0.531 / 0.545; ResNet-34 0.485 / 0.496 / 0.524 / 0.516 / 0.572.

Bias recalibration cannot change AUC (it only slides the threshold). On this data it does not help accuracy: Xception drops to 0.47-0.48 (its score ordering is slightly inverted, so a centred threshold sits on the wrong side), EfficientNet-B4 and ResNet-34 stay at 0.49-0.51.

All tables, including WiSE-style rows: `tables.md`. The `base/` prefix means original trunk; `adabn1000/` and `adabn_k/` prefixes will appear once those stages finish.

## Verdict (what the numbers show, nothing more)

- With 4-5 labelled images, none of the three CNNs finished so far moves off chance: every mean accuracy is within 0.485-0.506 and the seed std (0.005-0.035) is as large as any difference between methods. 4 images do not "program" these detectors.
- With 1,024 labelled images, the closed-form head reaches 0.55-0.58 accuracy and 0.57-0.62 AUC. That is above chance and clearly above k=4, but 13-17 points below our own trained model's 71-73% accuracy, and the AUC (0.57-0.62) is also well below its 0.78-0.82.
- The ceiling appears to be the frozen trunk: the penultimate features of these FF++-trained detectors hold only weak linear information about real vs fake on MegaDataset. A better head over those features can only recover what is there. AdaBN was measured only for ResNet-34 and raised the ceiling by about one to two seed standard deviations at k=1024 (0.581 to 0.590 accuracy), still far below 71-73%.
- WiSE-style interpolation with the original head helps little: at k=1024 it is below the pure closed-form head for all three models, and the original head is itself at or below chance on this data. At small k the differences are inside the seed noise (EfficientNet-B4 k=4: WiSE ridge a=0.25 is 0.503 vs ridge 0.491).
- Not shown: UnivFD and Effort edits, and AdaBN for Xception and EfficientNet-B4. No claim is made about them.

## Resuming

`python -I compare/weight_space.py run --model <name> --stages base,verify,ada1000,adak` (set `WSP_ADAK_KS=4,32` to limit AdaBN-k) after caching features with `... cache --model <name>`; then `... report`. Caches live in ~/wsp_cache (Test/Validation images and per-model features); wrap Test passes in the shared GPU lock (the script does this itself).

## References (arXiv pages for 1-5 were opened and checked; Effort and Ridge/LDA are cited from memory)

1. Li, Wang, Shi, Liu, Hou. Revisiting Batch Normalization for Practical Domain Adaptation (AdaBN), 2016. arXiv:1603.04779.
2. Zhang et al. Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification, ECCV 2022. arXiv:2207.09519 (cache-model / prototype classifiers from few-shot features).
3. Wortsman et al. Robust fine-tuning of zero-shot models (WiSE-FT), CVPR 2022. arXiv:2109.01903 (weight-space interpolation).
4. Ilharco et al. Editing Models with Task Arithmetic, ICLR 2023. arXiv:2212.04089.
5. Ojha, Li, Lee. Towards Universal Fake Image Detectors that Generalize Across Generative Models (UnivFD), CVPR 2023. arXiv:2302.10174.
6. Yan, Zhang, Yuan, Lyu, Wu. DeepfakeBench, 2023. arXiv:2307.01426 (Xception, EfficientNet-B4, CNN-Aug ResNet-34 detectors and FF++ c23 checkpoints).
7. Effort (ICML 2025, orthogonal subspace / SVD-residual CLIP detector), checkpoint effort_clip_L14_trainOn_FaceForensic.pth: cited from memory, paper not re-opened.
8. Classical background, from memory: ridge regression (Hoerl and Kennard 1970); regularised/shrinkage discriminant analysis (Friedman 1989; Ledoit and Wolf 2004); nearest-centroid prototypes (Snell et al. 2017).

## Files

`compare/weight_space.py` (single script: data, probe, cache, run, report); `probe_*.json`, `probe_tables.md`, `tune_*.json`, `rows_*.json`, `verify_*.json`, `summary.json`, `tables.md` in this folder.
