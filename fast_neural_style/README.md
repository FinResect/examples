# Ti60 Fast Neural Style

This fork prepares style-transfer models for Ti60F225I3 TinyML deployment.
See the Chinese [usage guide](使用说明.md) and [deployment guide](部署说明.md)
for training, calibration, operator audit and hardware integration requirements.

The default is `dw_c4_s` (`ti60_style_dw_c4_s_v1`): 7 depthwise 3x3 and
9 pointwise 1x1 convolutions, two nearest-neighbor 2x resizes, and no ADD.
Features use C4 except RGB input/output. Two stride-2 DW stages precede four
body DW/PW groups and one compression DW/PW group at quarter height/width,
then an extra PW, both resizes and a linear RGB PW. BatchNorm is folded before export.
With no spatial processing after upsampling, output consists of repeated 4x4 blocks.
Training does not clamp output; preview saving clips it to 0..255.
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
  --architecture dw_c4_s --image-size 256 --batch-size 4 --epochs 2 --accel
```

`--accel` selects CUDA. The first training run downloads pretrained VGG16 weights
unless they are already cached. Two epochs are a starting point, not a quality guarantee.
Checkpoints include architecture metadata and BN statistics, but no optimizer resume state.
Old InstanceNorm checkpoints and downloaded original pretrained models are incompatible.
Training uses existing VGG content/style losses only, with no QAT or teacher.
The default square size is 128. For VGA training, replace `--image-size 256` with
the paired `--image-height 480 --image-width 640`; square and rectangular options
are mutually exclusive. Legacy BN checkpoints/artifacts remain supported under
`ti60_style_bn_3x3_v1`; explicitly train them with `--architecture bn_3x3 --width 0.25`.

## Evaluation And Conversion

```bash
python neural_style/neural_style.py eval \
  --model outputs/models/ti60_epoch_2_TIMESTAMP.model \
  --content-image images/content-images/amber.jpg \
  --output-image outputs/preview.png
```

Evaluation defaults to checkpoint dimensions; add the paired
`--image-height 480 --image-width 640` for a VGA preview, or `--image-size 256`
for a square override, never both.

Use a separate conversion environment with `requirements-convert.txt`. Its version
ranges are candidate constraints and have not been validated together locally.

```bash
python -m pip install -r requirements-convert.txt
CUDA_VISIBLE_DEVICES=-1 TF_ENABLE_ONEDNN_OPTS=0 python script/model2tf_lite.py \
  --model outputs/models/ti60_epoch_2_TIMESTAMP.model \
  --height 480 --input-width 640 \
  --calib /data/calibration --calib-count 100 \
  --onnx outputs/style.onnx --saved-model outputs/saved_model \
  --int8-tflite outputs/style_int8.tflite
```

Conversion defaults to checkpoint dimensions, so the VGA override above is required
for a 128/256 checkpoint targeting VGA. The paired rectangular options are mutually
exclusive with `--size 256`. Converter `--width` is only a legacy BN consistency check.
The converter now supports the DW/rectangular interface, graph audit and compute
report, alongside the legacy BN contract. Actual conversion must still be verified
in the server environment; it was not run for this change.

The environment variables select CPU execution and disable oneDNN optimizations
for this conversion command and its children. A historical legacy-BN server run
completed conversion with both settings after a GPU parity failure; this neither
isolates the cause nor validates DW conversion. Keep trained weights unchanged;
strict parity tolerances remain the default. See the Chinese guide for the server command, report
inspection, preview paths and explanations of common TensorFlow log messages.

Conversion checks numerical parity, exports static NHWC INT8 IO and writes a JSON
operator/quantization report. The DW contract requires 7 DW, 9 Conv and two retained
2x resizes; a converter-fused single 4x resize must fail, without bypassing the audit.
Use 100–300 real calibration images. Review `status`, `blockers`, `audit`, `compute`,
`verification` and `hardware_proven`; compute must report per-layer weights/MACs
and target rate requirements, not ideal hardware FPS. Unexpected operators fail. Exit zero means
software checks passed, not hardware deployment proved. Hardware nearest-neighbor
resize, boundary pixel conversion and driver integration are still required to keep
RISC-V out of elementwise computation. Full-system resource use and timing require
FPGA synthesis and place-and-route. Original TinyML STANDARD integration needs an
Iris resolver update, strict hardware guards before DW CPU execution, synchronized
model_contract/profiler changes and Generator regeneration. Main-program model
enable flags are separate and are not switched on automatically.

Target B is 640x480 at 60 fps, unproven: 12,979,200 convolution MAC/frame requires
0.778752 GMAC/s for convolution alone and about 16.67 ms for the whole pipeline.
Benchmark STANDARD 4x2/8x2. Historical original 128x128 Rain at vendor 8x2, 100 MHz
took 354.55 ms Invoke (about 0.453 GMAC/s effective convolution throughput); this
does not predict DW performance. Historical 16x4 used 75,945 XLR and 281 RAM,
exceeding Ti60 capacity. See the deployment guide for quantization and padding rules.

## Experimental Folding Budget

Only for investigating later conversion stages, explicitly add
`--experimental-folding-atol 0.1` to the existing server command and use separate
experimental output paths. This is an example budget, not a validated accuracy
threshold. The value must be finite and greater than 0.01, in unclipped RGB output
units; relative tolerance remains 0.0001. No checkpoint or export weights change.

All three probes (content image, deterministic random input, zeros) retain their
original strict FP32 results in `parity`. Each also must pass the experimental
FP32 budget and a separate FP64 folding equivalence check using the original
strict tolerances. Budget/equivalence failures are collected across the probes
and block ONNX export; invalid shapes or non-finite results still abort immediately.
ONNX/SavedModel parity, INT8 conversion and graph audit are not relaxed.

Review `folding_experiment.probes` and `folding_experiment.strict_failures`.
Completed experiments always have status
`experimental_conversion_completed_hardware_unproven` and **exit code 3**, even
if their original strict folding checks also pass. They are not strict acceptance
or hardware proof. Failures still exit 1; invalid arguments exit 2. Without the
option, strict completion still exits 0. Check more content images before choosing
any acceptance budget; calibration images are not a multi-image parity test.

## Validation

For later server-side validation, from the repository root:

```bash
python -m unittest discover -s fast_neural_style -p test_ti60.py -v
```

Tests do not train or download weights. No scripts, commands shown above, or tests
were run for this change. The default local environment lacks training
and conversion dependencies. Reuse existing server environments for validation;
full conversion and hardware validation remain required. No external Iris workspace
was changed.
