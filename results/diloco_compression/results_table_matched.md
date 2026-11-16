| run | test acc | test AUC | val acc | val AUC | payload MB/worker | reduction vs bf16 | wire MB/worker | exchange s (mean/worker) | exposed s | hidden s | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bf16 | 70.5% | 0.762 | 84.7% | 0.926 | 690.2 | 1.0x | 1380.5 | 142 | 142 | 0 | 276 |
| int8_ef | 70.1% | 0.771 | 84.7% | 0.925 | 347.8 | 2.0x | 695.6 | 80 | 80 | 0 | 212 |
| bf16_overlap | 72.4% | 0.810 | 84.3% | 0.927 | 690.2 | 1.0x | 1380.5 | 147 | 30 | 117 | 158 |
| int8_ef_overlap | 72.4% | 0.810 | 84.3% | 0.927 | 347.8 | 2.0x | 695.6 | 92 | 16 | 76 | 142 |
