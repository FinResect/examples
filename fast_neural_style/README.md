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

## Compact Residual Candidate For VGA30

The quality reference is a `ti60_style_bn_3x3_v1`, width `0.25`, 128x128
checkpoint. The new opt-in `--architecture bn_c8_r2` uses a separate ID,
`ti60_style_bn_c8_r2_v1`. It preserves two residual blocks and spatial decoding:

```text
RGB -> Conv3x3/s2 C4 -> Conv3x3/s2 C8 -> 2 residual blocks (C8)
    -> nearest 2x -> Conv3x3 C4 + ReLU -> nearest 2x -> linear Conv3x3 RGB
```

All eight convolutions are ordinary 3x3, with two ADDs and two 2x resizes.
There is no Depthwise, InstanceNorm, reflection pad or output clamp. BN folds
into convolution. Unlike DW-C4-S, the decoder is not forced to repeat 4x4 blocks.
The existing default remains `dw_c4_s`; select the new architecture explicitly,
without `--width`. Train it from scratch in a separate directory; the larger
checkpoint is a visual reference, not a shape-compatible initialization.

| Logical quantity | BN width 0.25 | BN-C8-R2 |
|---|---:|---:|
| Convolution MAC/frame at 640x480 | 2,786,918,400 | 113,356,800 |
| INT8 convolution weight bytes, excluding bias/tables | 104,112 | 3,096 |
| Largest logical INT8 activation, including resize outputs | 4,915,200 bytes | 1,228,800 bytes |

The candidate has 3,243 trainable parameters (including BN), 204 bytes of
INT32 convolution biases after folding, and 6,045,696 convolution MAC/frame at
128x128. VGA30 requires **3.400704 GMAC/s for convolution alone**, not measured
throughput. Neither the ~24.6x logical MAC reduction nor the small weight count
proves 30 FPS, FPGA utilization, tensor-arena size or acceptable image quality.
The full-frame tensors still require a suitable external-memory/streaming design.
This change does not modify Iris firmware, RTL or board artifacts.

Add `--preview-image images/content-images/cobi.jpg --preview-interval 1000`
to training to save unfused FP32 eval previews at the training size and 480x640,
every 1,000 batches and at epoch end. Preprocessed inputs and previews go under
`<save-model-dir>/previews` unless `--preview-dir` is supplied. Previews do not
update BN statistics and restore all module training flags, including on failure.
Use the reference run's loss weights where available; checkpoint metadata does
not recover those settings. Do not change architecture and loss weights together
and then attribute the result only to architecture. Judge float previews first.
See the Chinese usage guide for the complete server command.

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

The historical DW 640x480/60 fps budget remains a comparison, not the current
compact candidate's VGA30 acceptance target: 12,979,200 convolution MAC/frame requires
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
With only this folding option, ONNX/SavedModel parity still uses the original FP32
reference and strict tolerances. INT8 conversion and graph audit are not relaxed.

Review `folding_experiment.probes` and `folding_experiment.strict_failures`.
Completed experiments always have status
`experimental_conversion_completed_hardware_unproven` and **exit code 3**, even
if their original strict folding checks also pass. They are not strict acceptance
or hardware proof. Failures still exit 1; invalid arguments exit 2. Without the
option, strict completion still exits 0. Check more content images before choosing
any acceptance budget; calibration images are not a multi-image parity test.

## Experimental FP64 Export Reference

If cross-runtime FP32 comparisons fail after folding, an additional explicit
`--experimental-fused-fp64-reference` option can be used with the folding budget.
This changes the experimental ONNX/SavedModel reference policy, not the numeric
tolerances or the exported model. It is not enabled by the folding option alone.

The converter copies the already-fused FP32 export parameters and upcasts that
copy to FP64, without re-fusing BN in FP64 or changing the export parameters.
All three inputs must first pass a strict comparison between the unfused model
and those export parameters, both evaluated in FP64. ONNX and SavedModel are then
compared to this validated reference with `atol=0.01, rtol=0.0001`; the original
FP32 comparisons remain in `parity`, including failures. The new comparisons and
reference validation are recorded in `fp64_reference`. Any new-reference failure
still blocks conversion. The same high-precision reference policy is used for
ONNX and SavedModel on every probe; the converter does not choose whichever
reference happens to pass.

The mode is `experimental_fused_fp64_reference`. Completion still exits 3, never
0, and is not original strict acceptance or hardware proof. INT8 graph auditing
is unchanged; `verification.float_reference` identifies the original fused FP32
baseline still used for INT8 quality metrics. Use separate experimental paths.
This option is unvalidated on the server checkpoint and may still fail; it does
not prove that the content-loss weight caused the original discrepancy.

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
