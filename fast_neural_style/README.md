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

## Scheme 3: Iris-TensorCore (104,112-Parameter 8x8 Hardware Unified Accelerator)

For maximum image fidelity and rich painterly artistic detail (such as Claude Monet's impressionist strokes), Scheme 3 uses the complete 16-convolution, 5-residual-block `TransformerNet(width=0.25)` model running on the custom 8x8 matrix accelerator:

- **104,112 parameters** (16 Convs, 5 ResBlocks, 2 Nearest-Neighbor Resizes).
- **8x8 Matrix Engine** (`iris_rain_conv.v`): 64 parallel INT8 MACs/cycle, on-chip 4-row caching, on-chip layer weight RAM.
- **Hardware Performance**: **30.05 FPS (31.77 ms Invoke)** on Ti60 @ 100MHz with **0 LSB error** vs TFLite golden reference.
- **Display**: Dual-phase display buffer (`iris_tinyml_panels`) outputting smooth, tear-free **60 FPS** on 720p/1080p HDMI.
- **Ti60 Resources**: 55,352 / 60,800 XLRs (91.0%), 246 / 256 RAMs (10 free), 143 / 160 DSPs (17 free), positive timing slack (+0.013 ns setup, +0.026 ns hold).

### Quickstart Workflow:
```bash
# 1. Environment check and venv setup
./script/setup_env.sh

# 2. Train any style model on GPU (e.g. Monet, Candy, etc.)
# Supports flat image directories (e.g. images/calib) or standard ImageFolder
./train.sh --dataset images/calib --style-image images/style-images/monet.jpg --name monet --epochs 2

# (train.sh will automatically train, quantize, clean temp files, and deploy to Iris/model_input/)
# If you train with python neural_style/neural_style.py train directly, export manually with:
# ./export_model.py --model model/<trained_model>.model --name <name>

# 3. Build RISC-V firmware and sync to board (in Iris repo)
cd ../../Iris
.script/build-firmware
.script/sync-iris jtag
```

### Essential Output Artifacts:
All temporary intermediate directories (`saved_model/`, `.onnx`) are automatically purged upon export.
Only the 5 required deployment artifacts are retained under `output/<name>/` and deployed to `Iris/model_input/`:
- `<name>.model`: PyTorch checkpoint with architecture metadata.
- `<name>_int8.tflite`: Static-shape fully quantized INT8 FlatBuffer model.
- `rain_backend_model.h`: 8x8 matrix accelerator packed weights and layer configs.
- `weights.bin`: Raw binary packed weights for the hardware matrix engine.
- `<name>_int8_model_data.cc` / `.h`: C array wrapper for embedding directly into firmware.

## Scheme C: Real-Time 720p60 Streaming Line-Buffer Distillation

To achieve 720p60 on Ti60F225 with 0 intermediate DDR round-trips, Scheme C distills a deep Teacher model (e.g. `TransformerNet`, 16 layers, 104k params) into an ultra-compact on-chip streaming line-buffer Student CNN (`StreamStudentNet`, 3 layers: Conv3s2 -> Conv3d2 -> Conv1s1 -> NN2x, 264 weights, 0 DDR bandwidth).

### 1. Training & Distillation
Run knowledge distillation using `distill_stream.py`:

```bash
# Option A: Distill from a trained Teacher model (.model)
python distill_stream.py \
  --teacher outputs/models/ti60_epoch_2.model \
  --dataset images/calib \
  --style-image images/style-images/monet.jpg \
  --candidate micro4_d2 \
  --steps 6000 \
  --batch-size 8 \
  --image-size 192 \
  --export-header /path/to/Iris/firmware/stream_style/stream_model.h

# Option B: Direct Style Distillation (if teacher is not pre-trained)
python distill_stream.py \
  --dataset images/calib \
  --style-image images/style-images/monet.jpg \
  --candidate micro4_d2 \
  --steps 6000 \
  --batch-size 8 \
  --export-header /path/to/Iris/firmware/stream_style/stream_model.h
```

The script trains the student network with:
- $L_1$ output distillation loss.
- Spatial gradient loss ($\nabla_x, \nabla_y$) to maintain crisp facial and edge contours.
- Shadow / contour intensity weighting ($1 + 3(1 - Y)$) to emphasize dark outline strokes.
- Automated calibration and INT8 quantization matching the Ti60 hardware contract.
- Direct emission of `stream_model.h` with exact 3-layer x 52 uint32 parameters.

## Validation

From the repository root:

```bash
python -m unittest discover -s fast_neural_style -p test_ti60.py -v
```

Tests do not train or download weights. Model tests skip when PyTorch is unavailable.
The current local environment lacks PyTorch and the conversion stack, so only the
preprocessing/parity helper tests have run; full model and conversion validation
must be performed in the server environment.
