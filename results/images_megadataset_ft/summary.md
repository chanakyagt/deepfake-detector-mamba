| Run (3,000 steps per worker, batch 64) | Inner steps H | Syncs | Payload sent per worker (MB) | Est. ring all-reduce traffic per worker (MB) | Test accuracy | Test AUC |
|---|---|---|---|---|---|---|
| 1 worker, Mamba | - | 0 | 0.0 | 0.0 | 89.2% | 0.936 |
| 4 workers, sync gradients every step | every step | 3000 | 69023.2 | 103534.9 | 88.8% | 0.932 |
| 4 workers, DiLoCo, H = 100 | 100 | 30 | 690.2 | 1035.3 | 87.5% | 0.923 |
| 4 workers, DiLoCo, H = 500 | 500 | 6 | 138.1 | 207.1 | 86.2% | 0.920 |
