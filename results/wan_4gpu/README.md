# DiLoCo over the public internet: 4 consumer GPUs (11 November 2026)

One real training run of the image detector (frozen EfficientNet-B0 features, 4-layer Mamba head, 11.5M trainable parameters) across four GPUs in different places, joined only by the internet through `dfmamba/hub.py` (one TCP port on one machine).

| Rank | GPU | Where | Training clips (its shard) |
|---|---|---|---|
| 0 | RTX 4060 Ti 8 GB | home PC, India, behind a home router | 35,001 |
| 1 | RTX 2060 12 GB | consumer GPU, remote site | 35,001 |
| 2 | RTX 3070 8 GB | consumer GPU, remote site | 35,000 |
| 3 | RTX 3060 12 GB | consumer GPU, remote site; also ran the hub | 35,000 |

Data: the 140,002-image MegaDataset training split as cached EfficientNet-B0 features, split four ways (`scripts/make_shards.py`). Validation: 39,428 images. Test: 10,905 images.

Settings: DiLoCo, H = 100 inner AdamW steps, batch 64 per worker, 3,000 steps, outer SGD with Nesterov momentum (lr 0.7, momentum 0.9), pseudo-gradients sent in bfloat16. Config: `diloco/` logs record it.

## Result

Note on validation: 13.5% of the Validation images (5,334 of 39,428) are byte-identical to Train images, so Validation is used only for checkpoint and threshold selection and is not quoted as a result.

| | |
|---|---|
| Test accuracy / AUC | **70.5% / 0.757** |
| Syncs | 30 (+1 start-up sync) |
| Data each worker uploaded | **690 MB** (23.0 MB per sync) |
| Wall-clock time | 869 s |
| Time rank 0 spent communicating | 400 s (about 12.9 s per sync) |

Per-worker timing (from `diloco/log_rank*.json`):

| Rank | GPU | Median time per 100-step segment | Waiting for others | Communicating |
|---|---|---|---|---|
| 0 | 4060 Ti (home) | 14.6 s | 15 s | 400 s |
| 1 | 2060 | 14.9 s | 509 s | 249 s |
| 2 | 3070 | 15.1 s | 579 s | 130 s |
| 3 | 3060 (hub host) | 15.1 s | 831 s | 5 s |

The home PC was the bottleneck: its connection to the hub is the slowest link, so the other three GPUs spent most of their non-compute time waiting for it. Its GPU was also shared with other experiments during part of the run.

## What the same run would cost with every-step syncing

Syncing after every step sends the same 23.0 MB per step instead of per 100 steps. A 40-step every-step run was started but stalled at start-up and was stopped, so this is a **projection from the measured DiLoCo sync cost**, not a measurement:

| | DiLoCo, H = 100 (measured) | Every-step sync (projected) |
|---|---|---|
| Syncs | 30 | 3,000 |
| Data uploaded per worker | 690 MB | about 69 GB |
| Communication time at 12.9 s per sync | 400 s | about 38,700 s (about 10.8 hours) |

On the single-PC simulation (`results/images_megadataset/`), every-step syncing reached 71.2% test accuracy against DiLoCo's 71.5%, so the 100x traffic saving costs no accuracy there.

## Notes

- Test accuracy here (70.5%) is a little below the single-PC DiLoCo run (71.5%, 4 workers on one GPU). The training data is the same, split 4 ways in both; one run each is not enough to say whether the 1-point gap is meaningful.
- Hub-to-home download speed depended heavily on where the hub was: about 0.1 MB/s from the 2060 and 3070 hosts and 2.7 MB/s from the 3060 host used, so hub placement matters as much as H.
- IP addresses are removed from `hub.log`.
