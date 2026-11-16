### xception

input [5, 3, 256, 256], penultimate feature [5, 2048], head W [2, 2048], b [2], BatchNorm layers 41, params 21,861,162

| layers | type | input | output | own params | params each |
|---|---|---|---|---|---|
| backbone.conv1 | Conv2d | [[5, 3, 256, 256]] | [5, 32, 127, 127] | {"weight": [32, 3, 3, 3]} | 864 |
| backbone.bn1 | BatchNorm2d | [[5, 32, 127, 127]] | [5, 32, 127, 127] | {"weight": [32], "bias": [32]} | 64 |
| backbone.relu | ReLU | [[5, 32, 127, 127]] | [5, 32, 127, 127] | {} | 0 |
| backbone.conv2 | Conv2d | [[5, 32, 127, 127]] | [5, 64, 125, 125] | {"weight": [64, 32, 3, 3]} | 18,432 |
| backbone.bn2 | BatchNorm2d | [[5, 64, 125, 125]] | [5, 64, 125, 125] | {"weight": [64], "bias": [64]} | 128 |
| backbone.block1 | Block | [[5, 64, 125, 125]] | [5, 128, 63, 63] | {} | 35,264 |
| backbone.block2 | Block | [[5, 128, 63, 63]] | [5, 256, 32, 32] | {} | 136,064 |
| backbone.block3 | Block | [[5, 256, 32, 32]] | [5, 728, 16, 16] | {} | 915,944 |
| backbone.block4 .. backbone.block11 (x8) | Block | [[5, 728, 16, 16]] | [5, 728, 16, 16] | {} | 1,613,976 |
| backbone.block12 | Block | [[5, 728, 16, 16]] | [5, 1024, 8, 8] | {} | 2,039,584 |
| backbone.conv3 | SeparableConv2d | [[5, 1024, 8, 8]] | [5, 1536, 8, 8] | {} | 1,582,080 |
| backbone.bn3 | BatchNorm2d | [[5, 1536, 8, 8]] | [5, 1536, 8, 8] | {"weight": [1536], "bias": [1536]} | 3,072 |
| backbone.conv4 | SeparableConv2d | [[5, 1536, 8, 8]] | [5, 2048, 8, 8] | {} | 3,159,552 |
| backbone.bn4 | BatchNorm2d | [[5, 2048, 8, 8]] | [5, 2048, 8, 8] | {"weight": [2048], "bias": [2048]} | 4,096 |
| backbone.last_linear | Linear | [[5, 2048]] | [5, 2] | {"weight": [2, 2048], "bias": [2]} | 4,098 |

### efficientnetb4

input [5, 3, 256, 256], penultimate feature [5, 1792], head W [2, 1792], b [2], BatchNorm layers 96, params 17,552,202

| layers | type | input | output | own params | params each |
|---|---|---|---|---|---|
| backbone.efficientnet._conv_stem | Conv2d | [[5, 3, 256, 256]] | [5, 48, 127, 127] | {"weight": [48, 3, 3, 3]} | 1,296 |
| backbone.efficientnet._bn0 | BatchNorm2d | [[5, 48, 127, 127]] | [5, 48, 127, 127] | {"weight": [48], "bias": [48]} | 96 |
| backbone.efficientnet._blocks.0 | MBConvBlock | [[5, 48, 127, 127]] | [5, 24, 127, 127] | {} | 2,940 |
| backbone.efficientnet._blocks.1 | MBConvBlock | [[5, 24, 127, 127]] | [5, 24, 127, 127] | {} | 1,206 |
| backbone.efficientnet._blocks.2 | MBConvBlock | [[5, 24, 127, 127]] | [5, 32, 63, 63] | {} | 11,878 |
| backbone.efficientnet._blocks.3 .. backbone.efficientnet._blocks.5 (x3) | MBConvBlock | [[5, 32, 63, 63]] | [5, 32, 63, 63] | {} | 18,120 |
| backbone.efficientnet._blocks.6 | MBConvBlock | [[5, 32, 63, 63]] | [5, 56, 32, 32] | {} | 25,848 |
| backbone.efficientnet._blocks.7 .. backbone.efficientnet._blocks.9 (x3) | MBConvBlock | [[5, 56, 32, 32]] | [5, 56, 32, 32] | {} | 57,246 |
| backbone.efficientnet._blocks.10 | MBConvBlock | [[5, 56, 32, 32]] | [5, 112, 16, 16] | {} | 70,798 |
| backbone.efficientnet._blocks.11 .. backbone.efficientnet._blocks.15 (x5) | MBConvBlock | [[5, 112, 16, 16]] | [5, 112, 16, 16] | {} | 197,820 |
| backbone.efficientnet._blocks.16 | MBConvBlock | [[5, 112, 16, 16]] | [5, 160, 16, 16] | {} | 240,924 |
| backbone.efficientnet._blocks.17 .. backbone.efficientnet._blocks.21 (x5) | MBConvBlock | [[5, 160, 16, 16]] | [5, 160, 16, 16] | {} | 413,160 |
| backbone.efficientnet._blocks.22 | MBConvBlock | [[5, 160, 16, 16]] | [5, 272, 8, 8] | {} | 520,904 |
| backbone.efficientnet._blocks.23 .. backbone.efficientnet._blocks.29 (x7) | MBConvBlock | [[5, 272, 8, 8]] | [5, 272, 8, 8] | {} | 1,159,332 |
| backbone.efficientnet._blocks.30 | MBConvBlock | [[5, 272, 8, 8]] | [5, 448, 8, 8] | {} | 1,420,804 |
| backbone.efficientnet._blocks.31 | MBConvBlock | [[5, 448, 8, 8]] | [5, 448, 8, 8] | {} | 3,049,200 |
| backbone.efficientnet._conv_head | Conv2dStaticSamePadding | [[5, 448, 8, 8]] | [5, 1792, 8, 8] | {"weight": [1792, 448, 1, 1]} | 802,816 |
| backbone.efficientnet._bn1 | BatchNorm2d | [[5, 1792, 8, 8]] | [5, 1792, 8, 8] | {"weight": [1792], "bias": [1792]} | 3,584 |
| backbone.efficientnet._swish | MemoryEfficientSwish | [[5, 48, 127, 127]] | [5, 48, 127, 127] | {} | 0 |
| backbone.last_layer | Linear | [[5, 1792]] | [5, 2] | {"weight": [2, 1792], "bias": [2]} | 3,586 |

### cnnaug_resnet34

input [5, 3, 256, 256], penultimate feature [5, 512], head W [2, 512], b [2], BatchNorm layers 36, params 21,285,698

| layers | type | input | output | own params | params each |
|---|---|---|---|---|---|
| backbone.resnet.0 | Conv2d | [[5, 3, 256, 256]] | [5, 64, 128, 128] | {"weight": [64, 3, 7, 7]} | 9,408 |
| backbone.resnet.1 | BatchNorm2d | [[5, 64, 128, 128]] | [5, 64, 128, 128] | {"weight": [64], "bias": [64]} | 128 |
| backbone.resnet.2 | ReLU | [[5, 64, 128, 128]] | [5, 64, 128, 128] | {} | 0 |
| backbone.resnet.3 | MaxPool2d | [[5, 64, 128, 128]] | [5, 64, 64, 64] | {} | 0 |
| backbone.resnet.4 | Sequential | [[5, 64, 64, 64]] | [5, 64, 64, 64] | {} | 221,952 |
| backbone.resnet.5 | Sequential | [[5, 64, 64, 64]] | [5, 128, 32, 32] | {} | 1,116,416 |
| backbone.resnet.6 | Sequential | [[5, 128, 32, 32]] | [5, 256, 16, 16] | {} | 6,822,400 |
| backbone.resnet.7 | Sequential | [[5, 256, 16, 16]] | [5, 512, 8, 8] | {} | 13,114,368 |
| backbone.avgpool | AdaptiveAvgPool2d | [[5, 512, 8, 8]] | [5, 512, 1, 1] | {} | 0 |
| backbone.fc | Linear | [[5, 512]] | [5, 2] | {"weight": [2, 512], "bias": [2]} | 1,026 |
