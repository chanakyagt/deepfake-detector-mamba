# Weight-space programming: an LLM tunes an ensemble head instead of backprop

Validation note: Validation figures in this document are used only for selecting edits, thresholds or combiner weights and are not headline results. Exact-copy overlap between Validation and Train (13.5%, 5,334 of 39,428 images) was measured on the full MegaDataset Validation split (`results/dataset_duplicates/dup.json`); the 6,000-image Validation subset used here was not checked separately. Quote Test numbers.

## What "weight space" means here

An LLM cannot hand-set millions of weights, so it searches a tiny weight space: the combiner that sits on top of several detectors.

Each detector i gives a probability p_i for "fake". We turn it into a score z_i = log(p_i / (1 - p_i)), clipped at +/-12. The combined output is

    logit = w_1*z_1 + ... + w_8*z_8 + b        (probability = sigmoid(logit))

That is 9 numbers (8 weights plus a bias). The 8 detectors are xception, efficientnetb4, cnnaug_resnet34, spsl, f3net, ucf, univfd and effort. The weights say how much the ensemble trusts each one, and a negative weight means "this detector points the wrong way on our data". The optional per-detector shift c_i is allowed by the design, but w_i*c_i is just a constant, so it is absorbed by b. Round 8 confirmed this: the AUC was identical and only accuracy moved with the threshold. The final vectors therefore do not use c.

This is a learned ensemble head in the sense of ensembling and weight averaging (model soups, Wortsman et al. 2022), but over output logits instead of network weights.

## Important caveats

- The 8 detectors were trained on FaceForensics++ and run zero-shot. Only the 9-number combiner was tuned, on 3,000 labelled validation images (the OPT split). No detector was retrained.
- Our fine-tuned EfficientNet-B0+Mamba model was not available when I ran (no probability files appeared in compare/results/backbones within the wait window), so it is not in the ensemble.
- The data shifts a lot. Validation AUCs are far higher than Test AUCs for the best detectors (univfd 0.777 on HOLD vs 0.627 on Test; effort 0.766 vs 0.658). Every method pays this price.

## Method

- Data: compare/results/zero_shot/probs/<model>_<split>.npz, aligned by image name; labels checked identical across detectors.
- Validation (6,000 images) split once with seed 0, stratified, into OPT (3,000) and HOLD (3,000). Test (10,905 images) is touched only by the single `final` command run.
- OPRO loop (Yang et al. 2023, arXiv:2309.03409): 20 rounds, 119 proposals. Each round I (the LLM) read the history of (params, OPT AUC, OPT acc, HOLD AUC), reasoned about it, and proposed 4-8 new vectors. All proposals and scores are in history.jsonl and the one-line rationale per round is in rounds.md. The evaluator is `compare/llm_weight_search.py eval`.
- Final LLM vector chosen by HOLD AUC (round 14, proposal 2).
- Baselines on the same OPT split:
  1. Best single detector, picked on OPT AUC (univfd).
  2. Equal-weight average of z.
  3. Backprop: logistic regression on z, 3,000 Adam steps in torch on OPT, with 9 parameters.
  4. Random search: 119 random vectors (the same number of evaluations as the LLM), chosen by HOLD AUC, the same rule as for the LLM.
- Test accuracy uses a fixed threshold of 0.5 on the combined probability.

## Results

| Method | #params | #evaluations | HOLD AUC | Test acc | Test AUC |
|---|---|---|---|---|---|
| LLM (OPRO) | 8 (7 weights + b; ucf weight 0) | 119 | 0.8422 | 0.6283 | 0.6883 |
| Backprop (logistic regression) | 9 | 3,000 gradient steps | 0.8407 | 0.6306 | 0.6912 |
| Random search | 9 | 119 | 0.7972 | 0.5924 | 0.6471 |
| Best single detector (univfd) | 0 tuned (picked among 8) | 8 | 0.7770 | 0.5465 | 0.6271 |
| Equal-weight average of z | 0 | 1 | 0.6728 | 0.5462 | 0.5612 |

OPT AUC (the data used for tuning): LLM 0.8339, backprop 0.8356, random 0.7949, best single 0.7681, equal 0.6794.

Final LLM parameters: cnnaug_resnet34 -0.10, efficientnetb4 -0.12, effort +0.40, f3net -0.10, spsl +0.10, univfd +0.19, xception +0.12, ucf 0, b = +0.65.

Backprop parameters: cnnaug_resnet34 -0.18, efficientnetb4 -0.11, effort +0.64, f3net -0.26, spsl +0.13, ucf +0.03, univfd +0.30, xception +0.19, b = +1.32.

Both methods agree on the structure: trust effort most, then univfd; give small positive weight to spsl and xception; give negative weight to cnnaug, efficientnetb4 and f3net. Those three have OPT AUC at or below 0.5 and subtracting them cancels noise shared with the correlated xception/ucf/efficientnetb4/f3net cluster. Ratios between weights matter for AUC; overall scale and b matter for accuracy at the 0.5 cutoff. The weights ending up with a similar ratio is not an accident: the LLM and the gradient are solving the same 9-dimensional problem.

## Verdict

- The LLM beat random search clearly (+0.041 Test AUC, +3.6 points Test acc) and beat the best single detector (+0.061 Test AUC, +8 points acc) and the naive average by a wide margin.
- The LLM did not beat backprop. Backprop is slightly better on Test (AUC 0.6912 vs 0.6883, acc 0.6306 vs 0.6283). A paired bootstrap on the Test set (500 resamples) puts the AUC difference at -0.003 with a 95% interval of about -0.004 to -0.002, so backprop is reliably but trivially better. The LLM was marginally better on HOLD AUC (0.8422 vs 0.8407), but that is the split used to pick the LLM vector, so it is not independent evidence.
- Cost: backprop reached its answer in 3,000 cheap CPU gradient steps with no judgement needed; the LLM needed 20 rounds of reasoning and 119 evaluations to land in the same place. With 9 parameters and a smooth convex loss, gradient descent is the right tool; the LLM only matches it.
- The 0.84 to 0.69 AUC gap between validation and Test is domain shift in the detectors, not a failure of either optimizer: the combiner generalises about as well as the best member allows. Re-weighting 3,000 labelled images does not fix detectors that were trained on a different distribution.
- Caveat on random search: it is a weak baseline here because uniform sampling in 9 dimensions rarely reaches the narrow high-AUC region. The LLM's advantage over it is the benefit of reading the history, which backprop gets for free from the gradient.

## Files

- compare/llm_weight_search.py (init, eval, baselines, final, show)
- history.jsonl: every LLM proposal with OPT AUC, OPT acc, HOLD AUC, HOLD acc
- random_history.jsonl: the 119 random-search evaluations
- rounds.md: one-line rationale per round
- final.json: Test scores and parameters for every method, plus per-detector Test metrics
- state.json: frozen detector list and the OPT/HOLD split

## References

- Yang et al. 2023, Large Language Models as Optimizers (OPRO), arXiv:2309.03409.
- Wortsman et al. 2022, Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time.
