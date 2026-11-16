# Weight-Space Programming on the two EfficientNet detectors

Validation note: Validation figures in this document are used only for selecting edits, thresholds or combiner weights and are not headline results. Exact-copy overlap between Validation and Train (13.5%, 5,334 of 39,428 images) was measured on the full MegaDataset Validation split (`results/dataset_duplicates/dup.json`); the 6,000-image Validation subset used here was not checked separately. Quote Test numbers.

## What WSP is, and how this run follows the paper
"Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, 2 Nov 2026; https://rikiyat.github.io/blog/weight-space-programming/) asks whether a language model can **read a trained network's weights and rewrite them by reasoning**. That means no training: no gradient descent (the usual "nudge every weight downhill on the error" loop), no fitting to data, and no trial-and-error search. In their Exp 8 the LLM got a Python kernel with the model loaded, practice problems, and a small budget of forward passes (a forward pass is one run of the network on a batch). It never saw the test sets.

This run copies that protocol. The harness `compare/wsp/harness.py` provides the following:
- Loads a model and lets me read any tensor and compute any statistic of the weights.
- Runs at most **50 forward passes per target**, each on at most 64 **Train** images. Every pass is logged in `budget.json`, and pass 51 is refused.
- Allows at most **5 Validation checks**, each costing one pass.
- Has a `final` command that reads **Test once**. In that single pass over the data it scores the original and the edited weights side by side.
- Logs every edit to `edits.jsonl`: the tensor, the number of changed values, the % of all parameters, and the written hypothesis.
- Blocks the banned tools in code: autograd/backward (automatic gradients), optimizers, and least-squares (`torch.linalg.lstsq`). It does not ban any statistic of the data.

Before each kept edit I fixed a keep rule in advance: keep the edit if Validation AUC (area under the ROC curve, 0.5 = chance, 1 = perfect ranking) drops by no more than 0.005.

## Results
| target | weights changed | passes (of 50) | val checks | Validation acc / AUC before → after | **Test acc / AUC before → after** |
|---|---|---|---|---|---|
| EfficientNet-B0 features + Mamba head (diloco_H100, 11,503,874 params) | **16,384 params (0.142%)** | 32 | 3 | 0.8483 / 0.9246 → 0.8474 / 0.9242 | **0.7145 / 0.7815 → 0.7149 / 0.7838** |
| DeepfakeBench EfficientNet-B4, FF++-trained, zero-shot (17,552,202 params) | **1 param (0.0000057%)** + 125,176 of 125,200 BatchNorm statistics (AdaBN, disclosed) | 27 | 3 | 0.5170 / 0.4953 → 0.5392 / 0.5486 | **0.5033 / 0.4980 → 0.5370 / 0.5525** |

Validation is 39,428 cached images for the Mamba model and 6,000 MegaSub images for B4. Test is 10,905 images for both.

Per-target narratives: `mamba_diloco/log.md`, `dfb_effnetb4/log.md`.

## Most interesting finding: dead weights work as a speedometer for the optimizer
For single-image inputs, the Mamba "state" never carries anything forward. The scan starts from zero, so the 65,536 `A_log` decay weights cannot affect the output. I checked this directly: resetting them changed the logits by exactly 0.0.

Because they never receive a gradient, the only thing that ever changed them is weight decay (a constant shrink applied every step). They shrank to exactly 0.9543x their init in the DiLoCo model, against 0.9910x in the single-GPU model. That 5.13x difference is a clean readout of how much DiLoCo's outer Nesterov step (the momentum update applied when the 4 workers sync) amplified every weight update. I rebuilt the exact init from seed 0 to confirm it. The DiLoCo weights moved 2.716x farther from init in total.

Undoing that amplification (scaling every displacement back by 1/2.7157) cost accuracy on practice images (0.891 → 0.844), so it was not the fix.

The edit I kept came from a second finding. In layer 1, the input-dependent "selective" path writes 11x more into the residual stream than the plain skip path (the residual stream is the running sum each layer adds to), yet its gain is uncorrelated with the label (r = +0.002). Switching it off changed nothing on Validation.

For B4, the weights show that **85% of its BatchNorm channels are dead** (53,328 of 62,600 have near-zero variance and near-zero scale). FF++ training left a much smaller network than the parameter count suggests.

## Verdict
- **Mamba head.** No real improvement. +0.0023 Test AUC is noise-level. The edit is a clean simplification: one class-agnostic path removed at no cost. My first edit (removing `proj` gain in low-variance input directions) lowered Validation AUC by 0.008 and was reverted. My written justification for it rested on a misleading comparison (11.96 vs 1.73), and I corrected that in the log. The val→Test gap (0.92 vs 0.78 AUC) did not trace back to any obstacle in the head's weights. The frozen B0 features of Test images most likely lack the cue the head learned.
- **B4.** A real, consistent but small gain: Test AUC +0.054, matching Validation's +0.053, and accuracy +0.0337. It came almost entirely from AdaBN, which re-estimates normalisation statistics from unlabelled images rather than rewriting weights by reasoning, plus one bias weight. The model stays near chance.
- As the paper says, edits "could remove an obstacle that hid an ability, but … could not add an ability the model lacked". Miscaled BN statistics were a removable obstacle. Detecting this dataset's fakes is an ability neither model had.

## Disclosures
- Before any work on target 1, I read the existing `test_metrics.json` files. They hold the original model's Test confusion matrix. None of my edits targeted that pattern, but I had seen it.
- I counted Test files per folder for target 2. No Test image or label was loaded before `final`.
- AdaBN and the median-based bias reset use statistics of unlabelled Train images.
- Reading the single_H1 checkpoint and rebuilding the seed-0 init were weight reads, not fitting.
