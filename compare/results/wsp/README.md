# Weight-Space Programming on downloaded deepfake detectors

Validation note: Validation figures in this document are used only for selecting edits, thresholds or combiner weights and are not headline results. Exact-copy overlap between Validation and Train (13.5%, 5,334 of 39,428 images) was measured on the full MegaDataset Validation split (`results/dataset_duplicates/dup.json`); the 6,000-image Validation subset used here was not checked separately. Quote Test numbers.

## What this is, in plain words

**Weight-Space Programming (WSP)** means a language model (here, me) opens a trained neural network, reads its numbers (its "weights": the values it learned during training), and rewrites a few of them by hand, using reasoning, so that the network does better at a new job. Nothing is trained:
- no gradients (the slope signals that training follows);
- no optimizer;
- no fitting of weights to data (least squares, logistic regression, LDA);
- no random or grid search;
- no trying many versions and keeping the best one.

Reference: R. Takehi, Teoh, Zhang, O. Khattab, "Sparks of Weight-Space Programming", MIT CSAIL, 2 Nov 2026, https://rikiyat.github.io/blog/weight-space-programming/. In the paper, editing 0.016% of a small language model's weights (removing one direction from 64 columns) raised MATH-500 from 7.9 to 48.1. The paper also states the limit: an edit can remove an obstacle that hides an ability the network already has, but it cannot add an ability.

**The question here:** can a downloaded state-of-the-art deepfake detector, trained on a different dataset (FaceForensics++, "FF++"), be hand-edited until it is as good on our MegaSub images as our own model? Our model scores 71.5-73.3% test accuracy and AUC 0.78-0.82.

Terms used below:
- **Accuracy:** the share of images labelled correctly at the 0.5 threshold.
- **AUC:** the chance that a random fake gets a higher fake score than a random real image. 0.5 is coin-flipping and 1.0 is perfect. AUC ignores where the threshold sits.
- **Margin:** the fake logit minus the real logit. Above 0 means "fake".
- **d':** the gap between the average fake and average real value of some quantity, in standard deviations. It measures how well that one quantity separates the classes.
- **BatchNorm:** a layer that rescales activations using a stored mean and variance, called "running statistics", remembered from the training data.
- **AdaBN:** re-measuring those stored statistics on new data. That counts as computing a statistic, not training, but it is disclosed here as AdaBN-like.

## Rules that were enforced

Harness: `compare/wsp/harness_sota.py`. It imports `compare/wsp/harness.py` for the pass budget, the edit log and the autograd ban, and adds the three targets.

- **Budget:** 50 forward passes per target, each on at most 64 labelled practice images from MegaSub/Train. Every pass is logged in `budget.json`, and pass 51 is refused.
- **Validation checks:** at most 5 per target, on the 6,000 images of MegaSub/Validation. Each check costs one pass.
- **Test:** the 10,905 images of MegaSub/Test can only be loaded by `final`, which refuses a second run. The original and edited weights are scored on the same single read.
- **Autograd and lstsq are disabled.** Every edit, including temporary ablations and reverts, is in `edits.jsonl` with the tensor, the number of values changed, the percentage and the hypothesis.

## Results (Test = one read per target)

### 1. Effort (ICML 2025; CLIP ViT-L/14 with rank-1 SVD-residual attention; FF++)

| | Validation acc | Validation AUC | Test acc | Test AUC |
|---|---|---|---|---|
| before (original weights) | 0.69933 | 0.76674 | 0.60018 | 0.65684 |
| after WSP | 0.71800 | 0.78713 | **0.63668** | **0.70048** |

- **Forward passes:** 15 of 50 (3 validation checks of 5).
- **Weights changed:** 2,050 = 0.00068% of the 303,378,530 parameters the forward pass uses. That is 0.00041% of all 504,606,722 registered parameters; the extra 201M are frozen training-only copies, `U_r/S_r/V_r`.
- **Kept edit:** one direction removed from the head's real-vs-fake vector, plus a mean-preserving bias shift.
- **Reverted:** a ×1.5 amplification of the 96 residual scalars. It lowered Validation to 0.69683 / 0.76548.

Details: `effort/log.md`.

### 2. UCF (ICCV 2023, DeepfakeBench, Xception-based; FF++ c23)

| | Validation acc | Validation AUC | Test acc | Test AUC |
|---|---|---|---|---|
| before | 0.52517 | 0.52049 | 0.50619 | 0.44940 |
| after WSP | 0.53433 | 0.54784 | **0.52187** | **0.51638** |

- **Forward passes:** 14 of 50 (2 validation checks of 5).
- **Weights changed:** 2 learnable parameters (0.0000043% of 46,838,430) plus 55,552 BatchNorm statistic values (an AdaBN-like re-estimation of all 41 `encoder_f` BatchNorm layers; 50% of the 111,104 buffer values).

Details: `ucf/log.md`.

### 3. Xception (DeepfakeBench baseline): did not run

The runs were stopped to free the GPU for higher-priority work. The Xception job was still queued behind the shared GPU lock and had not started. It was killed (PIDs 90295, 90301, 90302) before using any budget, so there is no `xception/` folder and no numbers.

### EfficientNet-B4 and our own model

These are reported in [EFFNET.md](EFFNET.md).

## The most interesting finding: what the weights were doing

**Effort's whole forgery skill sits in 96 numbers.** Effort freezes CLIP and lets each of its 96 attention projections learn just one extra rank-1 direction, scaled by one number `S_residual`. Those learned numbers are tiny: about 1e-9 to 2.5e-2, against CLIP's own singular values of 1.2 to 17.3. Yet setting all 96 to zero, which makes Effort plain CLIP plus its head, had large effects on the practice images:
- practice AUC fell from 0.797 to 0.686;
- d' fell from 1.27 to 0.79;
- the average fake margin fell from 1.61 to 0.26.

Turning the same 96 numbers up by 1.5× did not help on Validation, so their FF++ strength was already about right.

**The head was also reading a nuisance direction.** On our images, the second-largest direction of variation in Effort's 1024-d feature (PC1) moves the head's output a lot (0.54 of spread) while carrying no label information (d' = −0.08, within noise). Removing just that one direction from the head is the same kind of move as the paper's Qwen edit. It raised Validation accuracy by 1.87 points and AUC by 0.020, and Test accuracy by 3.65 points and AUC by 0.0436.

**UCF's problem is different and deeper.** Its deep BatchNorm layers see only about 0.06-0.7× the activation variance they expect from FF++, and its head input collapses to essentially one direction (97.6% of variance). Re-measuring the BatchNorm statistics fixes the scale, but the features underneath contain almost no information about MegaSub's fakes: Test AUC went from 0.449 to 0.516. That is the paper's limit in action. There was no hidden ability to uncover.

**A warning about Test.** Effort's original weights score 69.9% on Validation but only 60.0% on Test. Test is much harder for FF++-trained detectors than Validation is, so Validation numbers here flatter them.

## Verdict: can a downloaded detector, after WSP, stand in for our own model?

**Not at similar accuracy.** The best result is Effort after WSP: **63.67% Test accuracy, AUC 0.700**. That is 7.83 to 9.63 accuracy points and 0.0795 to 0.1195 AUC below our own model (71.5-73.3%, AUC 0.78-0.82).

The edit was real and generalised: +3.65 points, +0.0436 AUC, from changing 2,050 numbers without any training. But it cannot close the gap, because the gap is missing ability, not a hidden obstacle. On Validation, edited Effort (71.80%, AUC 0.787) looks like a match. Test shows it is not. UCF is near chance before and after. Xception was not run.

If a downloaded detector is wanted, edited Effort is the only candidate. It should be treated as a clearly weaker fallback, not a replacement.
