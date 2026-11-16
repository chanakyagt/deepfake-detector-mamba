# DiLoCo rerun on fine-tuned EfficientNet-B0 features

Features: frozen v2 trunk (EfficientNet-B0 fine-tuned end to end on the full 140k training set at 256 px, RTX 3090 retrain, test 89.62% / AUC 0.958 as a stand-alone classifier), extracted for train / validation / test of the full MegaDataset (1280-dim, one frame per image). Head: configs/images.yaml (d_model 512, 4 Mamba layers), 3000 steps per worker, batch 64, 4 simulated workers on one GPU. Single run per row. Commands are those of scripts/run_image_experiment.sh (single, allreduce H1, diloco H100, diloco H500). Logs and metrics: this folder.

| Run | Inner steps H | Syncs | Payload sent per worker (MB) | Est. ring all-reduce traffic per worker (MB) | Test accuracy | Test AUC |
|---|---|---|---|---|---|---|
| 1 worker, Mamba | - | 0 | 0.0 | 0.0 | 89.2% | 0.936 |
| 4 workers, sync gradients every step | every step | 3000 | 69023.2 | 103534.9 | 88.8% | 0.932 |
| 4 workers, DiLoCo, H = 100 | 100 | 30 | 690.2 | 1035.3 | 87.5% | 0.923 |
| 4 workers, DiLoCo, H = 500 | 500 | 6 | 138.1 | 207.1 | 86.2% | 0.920 |

Reading it. With these better features, DiLoCo does NOT fully match every-step synchronisation: H = 100 is 1.3 points of accuracy and 0.009 AUC below the every-step run, and H = 500 is 2.6 points and 0.012 AUC below, in exchange for 100 times and 500 times less traffic. Every-step sync is itself 0.4 points below the single worker. These are single runs with no repeated seeds, so differences under about 1 point could be noise, but the H = 100 and H = 500 gaps are in the same direction and grow with H. The head on frozen fine-tuned features (89.2%, AUC 0.936) is also below the end-to-end fine-tuned classifier (89.62%, AUC 0.958), because the fine-tuned trunk has seen the training images and its features on them are easier than on the test set.
