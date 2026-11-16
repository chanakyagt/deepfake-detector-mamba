# Small Mamba heads on v2 fine-tuned EfficientNet-B0 features

Features: ~/work/mega_feats_ft (frozen v2 trunk, 1280-dim, full MegaDataset), configs/images.yaml with the listed d_model / n_layers, single worker, 3000 steps, batch 64. n_layers = 0 is an MLP only. Test = full 10,905-image test split. Input is one frame, so the Mamba layers act as extra MLP-like capacity here.

| d_model | Mamba layers | Trainable params | Test acc | Test AUC | Best val acc |
|---|---|---|---|---|---|
| 256 | 2 | 1,729,666 | 88.8% | 0.936 | 98.6% |
| 128 | 1 | 350,786 | 88.8% | 0.927 | 98.6% |
| 128 | 2 | 528,962 | 88.7% | 0.928 | 98.6% |
| 64 | 1 | 132,386 | 88.7% | 0.933 | 98.6% |
| 64 | 2 | 180,514 | 88.7% | 0.945 | 98.6% |
| 64 | 0 | 84,258 | 88.5% | 0.931 | 98.6% |

## Hardware target (d128 L2)

Head: d_model 128, 2 Mamba layers, trained on ~/work/mega_feats_ft (run ftsweep_d128_n2). Every nn.Linear weight in the head (516,224 of 528,962 parameters; this includes the input projection, the Mamba in/x/dt/out projections and the classifier layers) is fake-quantised with a symmetric per-output-channel scale (one scale per weight row, group size = full row, round to nearest, no clipping search). Biases, the depthwise conv, A_log, D and LayerNorm stay fp32, and activations stay fp32. Evaluated on the full 10,905-image test split; flips are counted at threshold 0.5 against the fp32 predictions.

| Precision | Test acc | Test AUC | Prediction flips vs fp32 |
|---|---|---|---|
| fp32 | 88.74% | 0.9283 | 0 of 10905 |
| int8 | 88.74% | 0.9283 | 0 of 10905 |
| int4 | 88.77% | 0.9273 | 5 of 10905 |
