# Ti60 Fast Neural Style

This fork prepares style-transfer models for Ti60F225I3 TinyML deployment.
See [the Chinese training and deployment guide](使用说明.md) for the full workflow,
calibration, operator audit, resource estimates and hardware integration requirements.

The default model uses width 0.25 (8/16/32 channels), 128x128 RGB images,
sixteen 3x3 convolutions, five residual blocks, BatchNorm folded before export,
zero SAME padding and two nearest-neighbor upsampling stages.
The perceptual loss follows [Perceptual Losses for Real-Time Style Transfer](https://arxiv.org/abs/1603.08155).
VGG16 is used only during training and is not exported.

## Training

Run from this directory. Install a matching CUDA PyTorch/torchvision pair on the
server, then install `requirements.txt`. The dataset uses ImageFolder subdirectories.

```bash
python -m pip install -r requirements.txt
python neural_style/neural_style.py train \
  --dataset /data/content \
  --style-image images/style-images/mosaic.jpg \
  --save-model-dir outputs/models \
  --width 0.25 --image-size 128 --batch-size 4 --epochs 2 --accel
```

`--accel` selects CUDA. The first training run downloads pretrained VGG16 weights
unless they are already cached. Two epochs are a starting point, not a quality guarantee.
Checkpoints include architecture metadata and BN statistics, but no optimizer resume state.
Old InstanceNorm checkpoints and downloaded original pretrained models are incompatible.

## Evaluation And Conversion

```bash
python neural_style/neural_style.py eval \
  --model outputs/models/ti60_epoch_2_TIMESTAMP.model \
  --content-image images/content-images/amber.jpg \
  --output-image outputs/preview.png
```

Use a separate conversion environment with `requirements-convert.txt`. Its version
ranges are candidate constraints and have not been validated together locally.

```bash
python -m pip install -r requirements-convert.txt
CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 python script/model2tf_lite.py \
  --model outputs/models/ti60_epoch_2_TIMESTAMP.model \
  --calib /data/calibration --calib-count 100 \
  --onnx outputs/style.onnx --saved-model outputs/saved_model \
  --int8-tflite outputs/style_int8.tflite
```

The environment variables select CPU execution and disable oneDNN optimizations
for this conversion command and its children. This avoids the GPU/CPU parity
failure observed on the training server without changing trained weights or
loosening tolerances. See the Chinese guide for the server command, report
inspection, preview paths and explanations of common TensorFlow log messages.

Conversion checks numerical parity, exports static NHWC INT8 IO and writes a JSON
operator/quantization report. Unexpected operators fail the audit. Exit zero means
software checks passed, not hardware deployment proved. Hardware nearest-neighbor
resize, boundary pixel conversion and driver integration are still required to keep
RISC-V out of elementwise computation. Full-system resource use and timing require
FPGA synthesis and place-and-route.

## Validation

From the repository root:

```bash
python -m unittest discover -s fast_neural_style -p test_ti60.py -v
```

Tests do not train or download weights. Model tests skip when PyTorch is unavailable.
The current local environment lacks PyTorch and the conversion stack, so only the
preprocessing/parity helper tests have run; full model and conversion validation
must be performed in the server environment.
