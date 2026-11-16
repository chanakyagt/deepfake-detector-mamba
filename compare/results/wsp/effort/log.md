# WSP log: Effort (CLIP ViT-L/14 + SVD-residual attention, FF++ checkpoint)

Harness: `compare/wsp/harness_sota.py` (wraps the `compare/wsp/harness.py` budget, edit log and autograd ban). Forward passes in bf16 autocast, weights kept in fp32.
Practice images come from MegaSub/Train only. Test was read once, by `final`.

## Model facts read from the weights (no forward passes)

- 504,606,722 registered parameters, of which 303,378,530 are used by the forward pass. The other 201,228,192 are the 288 `U_r / S_r / V_r` tensors: frozen copies of CLIP's top-1023 singular directions that only the training-time orthogonality loss uses. They are absent from the checkpoint (the 288 "missing keys") and are rebuilt from the CLIP weights at construction, so they do not affect any score. All percentages below use the 303,378,530 forward-used parameters as the denominator.
- Head: `Linear(1024, 2)`, row norms 0.621 / 0.603, bias [0.0184, 0.0012]. The fake-minus-real direction `wd = W[1] - W[0]` has norm 0.932.
- Each of the 96 attention projections (24 layers x q/k/v/out) carries one learned rank-1 residual `U_residual (1024x1) * S_residual * V_residual (1x1024)` added to the frozen CLIP weight. The learned `S_residual` values are tiny: for example 0.0042 (layer 1 out_proj), -0.0246 (layer 23 v_proj), and about 1e-9 to 1e-7 in layer 0. CLIP's main singular values are 1.2 to 17.3.

## Passes

| pass | kind | what |
|---|---|---|
| 1 | val | baseline, original weights: acc 0.69933, AUC 0.76674 (tp 2043, tn 2153, fp 847, fn 957) |
| 2-5 | practice | 4 x 64 balanced Train images; recorded the pooled 1024-d head input and the raw CLS |
| 6-9 | practice | temporary ablation (not kept): all 96 `S_residual` set to 0 (= plain CLIP attention), same images as passes 2-5 |
| 10 | val | after edit 1: acc 0.71800, AUC 0.78713 (tp 2164, tn 2144, fp 856, fn 836) |
| 11-14 | practice | edit 2 test: `S_residual` x1.5, same images as passes 2-5 |
| 15 | val | after edit 2: acc 0.69683, AUC 0.76548, so edit 2 was reverted |
| final | Test | one read, original and edited weights scored batch by batch |

15 of 50 passes used, 3 of 5 validation checks used.

## What the statistics showed

Principal-component analysis (PCA: finding the directions in which the 256 practice feature vectors vary most) of the pooled feature, on practice images only:

| direction | variance | head weight on it (`wd . v`) | margin spread it causes | real/fake separation d' |
|---|---|---|---|---|
| PC0 | 46.4 | -0.199 | 1.355 | -1.073 (real signal) |
| PC1 | 27.4 | +0.103 | 0.538 | -0.076 (no signal) |
| PC2 | 27.1 | -0.121 | 0.631 | -0.505 (real signal) |

(d' is the gap between the fake and real averages, measured in standard deviations. Its noise level with 256 images is about 0.125.)

PC1 adds about 0.54 of spread to the logit margin while carrying no information about the label. The FF++ head reads a direction that, on this data, is pure nuisance.

Ablation (passes 6-9): switching off the 96 residual scalars moved the practice AUC from 0.797 to 0.686, d' from 1.271 to 0.791, and the mean fake margin from 1.615 to 0.256. Roughly 96 numbers out of 303 million carry most of what Effort learned about forgeries.

## Edits

1. **Kept.** Remove PC1 from the head's fake-minus-real direction (`W0 += c/2 v1`, `W1 -= c/2 v1`, `c = 0.10277`), so the sum `W0 + W1` is unchanged. Move the bias by `c * (v1 . mu) = -0.62169` so that the average margin over the practice images is unchanged (no threshold move). 2,048 head weights + 2 biases = 2,050 values = 0.00068% of forward-used parameters.
2. **Reverted.** Multiply all 96 `S_residual` by 1.5, a factor fixed in advance, plus a label-free bias recentre. Hypothesis: weight decay left the forgery residuals under-strength. Practice AUC rose only from 0.83783 to 0.83868 (noise), and validation fell to 0.69683 / 0.76548. Reverted to the edit-1 state, which had already been validated at pass 10. The edit log keeps both the edit and the revert.

The temporary ablation is logged in `edits.jsonl` as two `temporary_ablation_on/off` records and in `ablations.jsonl`.

## Result

| split | original acc | original AUC | edited acc | edited AUC |
|---|---|---|---|---|
| Validation (6,000) | 0.69933 | 0.76674 | 0.71800 | 0.78713 |
| Test (10,905) | 0.60018 | 0.65684 | 0.63668 | 0.70048 |

Changed: 2,050 values (0.00068% of the 303,378,530 forward-used parameters; 0.00041% of all 504,606,722 registered).

Effort's Test score is about 10 points below its Validation score even with the original weights, so Test contains material that Effort's FF++ features handle much worse than Validation. The edit's gain carried over to Test (+3.65 accuracy points, +0.0436 AUC).

## Notes

- PC1 was chosen from the practice images using their labels, but only to read d' as a diagnostic. No weight was fitted to activations: the edit is a projection that removes one direction, and its size is fixed by the existing head weights.
- The bias moves only to keep the practice-mean margin unchanged. That uses the practice feature mean, a label-free statistic.
- Two hypotheses were checked on validation in sequence, and the failed one was reverted. That is the extent of any selection.
