# WSP log: UCF (DeepfakeBench, Xception-based, FF++ c23 checkpoint)

Harness: `compare/wsp/harness_sota.py`. Input 256x256, mean/std 0.5. The checkpoint loaded with 0 missing and 0 unexpected keys. 46,838,430 parameters plus 111,104 floating-point buffer values (BatchNorm running means and variances).

At inference UCF uses only the forgery encoder `encoder_f`, then `block_sha`, then the shared head `head_sha` (pool to 256-d, `Linear 256->512`, LeakyReLU, `Linear 512->2`). The content encoder `encoder_c` is computed but does not affect the output.

## Passes

| pass | kind | what |
|---|---|---|
| 1 | val | baseline: acc 0.52517, AUC 0.52049 (tp 652, tn 2499, fp 501, fn 2348): calls almost everything real |
| 2-5 | practice | 4 x 64 balanced Train images; head input feature and BatchNorm input moments |
| 6-9 | practice | same images, BatchNorm in batch-statistics mode (momentum 0, so stored stats untouched); recorded the cascaded BN input moments |
| 10-13 | practice | same images after the BN edit, eval mode; label-free margin statistics |
| 14 | val | after edits: acc 0.53433, AUC 0.54784 |
| final | Test | one read |

14 of 50 passes, 2 of 5 validation checks.

## What the statistics showed

- Practice AUC of the original model was 0.561. The 256-d head input has 97.6% of its variance in a single direction, and no feature dimension separates real from fake by more than |d'| = 0.458 (d' = gap between the fake and real averages in standard deviations).
- BatchNorm mismatch: from block 8 onward, the variance entering each `encoder_f` BatchNorm on MegaSub is only 0.06x to 0.73x the FF++ running variance (median ratio per layer) (median ratio 0.058 at `bn3`, about 0.22 in blocks 9-11). The mean shift is small (0.04 to 0.63 standard deviations). Every deep layer is therefore normalised with the wrong scale.

## Edits

1. **AdaBN-like re-estimation (disclosed).** The running mean and variance of all 41 `encoder_f` BatchNorm layers were replaced by the moments of the 256 practice images, measured in a single cascaded batch-statistics pass so that each layer sees inputs already renormalised upstream. These are statistics, not a fit, and no labels were used. That is 82 buffer tensors, 55,552 buffer values.
2. **Label-free threshold recentre.** After edit 1, the median practice margin was +0.00377. The `head_sha.mlp.2` bias was moved so that this median becomes the threshold (assumes about 50% fakes; labels not used). 2 parameters.

Under batch statistics the practice AUC rose to 0.635 (0.630 after freezing the stats), but validation AUC moved only from 0.520 to 0.548.

## Result

| split | original acc | original AUC | edited acc | edited AUC |
|---|---|---|---|---|
| Validation (6,000) | 0.52517 | 0.52049 | 0.53433 | 0.54784 |
| Test (10,905) | 0.50619 | 0.44940 | 0.52187 | 0.51638 |

Changed: 2 learnable parameters (0.0000043% of 46,838,430) plus 55,552 BatchNorm statistic values (50% of the 111,104 buffer values; they are statistics, not trained weights).

## Verdict

UCF's FF++ forgery features carry almost no signal about MegaSub's fakes. Fixing the BatchNorm scale removed one obstacle, but the features underneath have no ability to uncover. This is the paper's stated limit: an edit can remove an obstacle but cannot add an ability. No further validation checks were spent.
