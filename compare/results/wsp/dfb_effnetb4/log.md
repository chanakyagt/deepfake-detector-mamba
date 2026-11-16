# WSP log: DeepfakeBench EfficientNet-B4 (FF++ c23 checkpoint, used zero-shot)

Model: `build_dfb('efficientnetb4', ...)`. 17,552,202 parameters plus 125,200 BatchNorm buffer values; the checkpoint loaded with 0 missing and 0 unexpected keys. Input is 256x256 with mean/std 0.5. Data is `~/data/MegaSub`: practice images come from Train (20k real + 20k fake), checks use Validation (3k + 3k). Budget used: **27 of 50 forward passes, 3 of 5 validation checks**, plus 1 final Test read. Peak GPU memory was 2.02 GB, so the 3 GB lock rule did not apply.

Disclosure: during the first look at the environment I counted the files in the Test folders (5,413 real, 5,492 fake). No Test image or label was loaded before `final`.

## 0. Baseline (pass 1, val check 1)
Validation: acc 0.5170, AUC 0.4953. That is chance. The mean p_fake is 0.344 on reals and 0.354 on fakes.
Survey (passes 2-5, 256 Train practice images): batch AUC 0.621, with a logit difference (fake minus real) of -0.72 on reals and -0.02 on fakes. The FF++ model has a faint signal on this dataset's Train split and none on its Validation split.

## 1. Finding: 85% of the BatchNorm channels are dead
I counted channels where both running_var < 1e-6 and |gamma| < 1e-3: **53,328 of 62,600**. Whole expansion layers in the last blocks are gone (block 31: 2,687 of 2,688; blocks 27-29: more than 99%). In the final head BN, 35.3% of channels are dead. The head weights on those dead channels add up to an absolute sum of 0.115, against 130.3 over all channels. The deep part of the network therefore runs mostly through its skip connections and a small set of live channels. Most of its nominal capacity was switched off during FF++ training, which leaves very little for an edit to "unlock".

## 2. Edit E1: AdaBN, re-estimating every BatchNorm running statistic. KEPT
- **Hypothesis.** The 96 BN layers normalise with means and variances measured on FF++ video frames. On this data they are miscentred and miscaled, so every layer downstream sees shifted inputs.
- **Evidence (passes 2-5, eval mode).** The mean |z|-shift of BN inputs against the stored statistics is at most 0.35 per layer. Early layers show variance ratios of 1.1 to 1.8. The largest gap is in the final head BN `_bn1`: mean |z| 0.305, and the observed variance is only about 0.13 to 0.19 of the stored one (median 0.00058 vs 0.0031). The features reaching the classifier are compressed relative to what it was trained on.
- **Edit.** Over 16 passes x 64 unlabelled Train images, with BN layers normalising by batch statistics (momentum 0, so stored statistics stay fixed during collection), I pooled each BN layer's input mean and variance and wrote them into running_mean and running_var. This is an **AdaBN-like statistic re-estimation (disclosed)**: labels are not used and no weights are fitted. Changed: **125,176 buffer values, 0 parameters**.
- **Effect.** On the pass-2 batch, batch AUC went from 0.621 to 0.643 and accuracy was 0.547. Validation (pass 23): **AUC 0.4953 -> 0.5486**, but accuracy 0.5128, because mean p_fake rose to 0.766 on reals and 0.794 on fakes: almost everything is now called fake.

## 3. Edit E2: put the decision threshold back between the classes. KEPT
- **Hypothesis.** After AdaBN the ranking improved but the threshold is wrong. Train is 50% fake by construction, so the median logit difference over unlabelled Train images should be the threshold.
- **Evidence (passes 24-26, 192 unlabelled images).** Median logit difference 3.9504, mean 4.1557. 84.9% of the images are on the fake side.
- **Edit.** Fake bias -= 3.9504. Changed: **1 parameter**.
- **Effect (pass 27).** Validation acc **0.5392**, AUC 0.5486 (unchanged, as it must be). Mean p_fake: 0.411 on reals, 0.481 on fakes.

I stopped there. The remaining ideas, such as picking penultimate channels by their label signal on practice batches, would amount to fitting to labels, which the protocol bans.

## 4. Final (one Test read, 10,905 images; original and edited weights scored in the same pass)
| weights | Test acc | Test AUC | tp | tn | fp | fn |
|---|---|---|---|---|---|---|
| original | 0.5033 | 0.4980 | 1647 | 3842 | 1571 | 3845 |
| edited (AdaBN + bias) | 0.5370 | 0.5525 | 1884 | 3972 | 1441 | 3608 |

Test moved by the same amount as Validation (AUC +0.054 on Test, +0.053 on Validation). The gain is consistent, but small in absolute terms.

**Verdict.** AdaBN removed a real obstacle, the miscaled normalisation statistics, and recovered a small amount of ranking signal on Validation (AUC +0.053). The bias edit made that signal usable as accuracy. The resulting detector is still close to chance. With 85% of its BN channels dead and no zero-shot signal on this domain, the FF++ model lacks the ability, and changing the weights cannot add it.
