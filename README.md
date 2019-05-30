# SRM-Tensorflow
Simple Tensorflow implementation of "SRM : A Style-based Recalibration Module for Convolutional Neural Networks" | [paper](https://arxiv.org/abs/1903.10829)

<div align="center">
  <img src="./assets/teaser.png">
</div>

## Usage
```python
from SRM import SRM_block

x = SRM_block(x, channels, use_bias=False, is_training=is_training, scope='srm_block')

```

## Comparison
<img src = './assets/compare.png' width = '500px' height = '500px'>

## Results
### Classification
<img src = './assets/result_classification.png'>

### Style Transfer
<img src = './assets/result_transfer.png'>

### Reference
* [SRMNet-Pytorch](https://github.com/EvgenyKashin/SRMnet)

## Author
Junho Kim
