"""Convert metadata-bearing compact BN, DW or legacy BN models to Ti60 INT8 candidates.

Every invocation runs every stage. The old skip flags are intentionally removed.
An operator audit is a necessary condition, not proof of hardware deployment.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STRICT_PARITY_ATOL = 0.01
STRICT_PARITY_RTOL = 0.0001


def record_stage(report, stage):
    report["stage"] = stage
    print(f"[stage] {stage}", file=sys.stderr, flush=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Metadata-bearing unfused training checkpoint")
    p.add_argument("--width", type=float, help="Legacy BN width consistency check only")
    p.add_argument("--size", type=int, help="Static square size; defaults to checkpoint image_size; multiple of 4")
    p.add_argument("--height", type=int, help="Static input height; supply with --input-width")
    p.add_argument("--input-width", type=int, help="Static input width; supply with --height")
    p.add_argument("--onnx", default="style.onnx")
    p.add_argument("--saved-model", default="saved_model", help="Parent for a fresh, retained run directory")
    p.add_argument("--int8-tflite", default="style_int8.tflite")
    p.add_argument("--report", help="JSON report; defaults to <int8-tflite>.report.json")
    p.add_argument("--calib", default="calib", help="Recursive RGB jpg/jpeg/png calibration directory")
    p.add_argument("--calib-count", type=int, default=100, help="Maximum calibration images (default: 100)")
    p.add_argument("--content", help="Verification image; defaults to first calibration image")
    p.add_argument("--output-image", help="Clipped verification PNG; defaults to <int8-tflite>.preview.png")
    p.add_argument("--experimental-folding-atol", type=float,
                   help="Experimental FP32 BN-folding absolute tolerance in raw RGB units; "
                        "must exceed 0.01. Other checks stay strict; completion exits 3, not 0")
    p.add_argument("--experimental-fused-fp64-reference", action="store_true",
                   help="Experimental ONNX/SavedModel reference from the exported FP32 weights "
                        "evaluated in FP64, with unchanged strict tolerances; requires a folding budget")
    args = p.parse_args()
    if (args.height is None) != (args.input_width is None):
        p.error("--height and --input-width must be supplied together")
    if args.size is not None and args.height is not None:
        p.error("--size is mutually exclusive with --height/--input-width")
    if args.size is not None or args.height is not None:
        try:
            dimensions(args.size if args.size is not None else (args.height, args.input_width))
        except ValueError as error:
            p.error(str(error))
    if args.calib_count <= 0:
        p.error("--calib-count must be positive")
    if args.experimental_folding_atol is not None and (
            not math.isfinite(args.experimental_folding_atol) or
            args.experimental_folding_atol <= STRICT_PARITY_ATOL):
        p.error("--experimental-folding-atol must be finite and greater than 0.01")
    if args.experimental_fused_fp64_reference and args.experimental_folding_atol is None:
        p.error("--experimental-fused-fp64-reference requires --experimental-folding-atol")
    return args


def preprocess(path, size):
    """Shared RGB bilinear aspect-fill and center crop; NHWC float32 0..255."""
    import numpy as np
    from PIL import Image
    from neural_style.preprocessing import resize_crop_image

    with Image.open(path) as source:
        image = resize_crop_image(source, size)
        pixels = np.asarray(image, dtype=np.float32)
    return pixels[None]


BN_ARCHITECTURE = "ti60_style_bn_3x3_v1"
DW_ARCHITECTURE = "ti60_style_dw_c4_s_v1"
COMPACT_ARCHITECTURE = "ti60_style_bn_c8_r2_v1"


def dimensions(size):
    height, width = (size, size) if isinstance(size, int) else size
    if any(type(n) is not int or n < 16 or n % 4 for n in (height, width)):
        raise ValueError("Dimensions must be integers >=16 and divisible by four")
    return height, width


def expected_layers(architecture, height, width):
    """Ordered (opcode, H, W, Cin, Cout, kernel, stride) deployment contract."""
    if architecture == COMPACT_ARCHITECTURE:
        quarter_height, quarter_width = height // 4, width // 4
        layers = [("CONV_2D", height // 2, width // 2, 3, 4, 3, 2),
                  ("CONV_2D", quarter_height, quarter_width, 4, 8, 3, 2)]
        for _ in range(2):
            layers.extend([("CONV_2D", quarter_height, quarter_width, 8, 8, 3, 1),
                           ("CONV_2D", quarter_height, quarter_width, 8, 8, 3, 1),
                           ("ADD", quarter_height, quarter_width, 8, 8, 0, 1)])
        return layers + [("RESIZE_NEAREST_NEIGHBOR", height // 2, width // 2, 8, 8, 0, 2),
                         ("CONV_2D", height // 2, width // 2, 8, 4, 3, 1),
                         ("RESIZE_NEAREST_NEIGHBOR", height, width, 4, 4, 0, 2),
                         ("CONV_2D", height, width, 4, 3, 3, 1)]
    if architecture == DW_ARCHITECTURE:
        layers = []
        h, w = height, width
        for ci, stride in ((3, 2), (4, 2)) + ((4, 1),) * 5:
            h, w = h // stride, w // stride
            layers.extend([("DEPTHWISE_CONV_2D", h, w, ci, ci, 3, stride),
                           ("CONV_2D", h, w, ci, 4, 1, 1)])
        return layers + [("CONV_2D", h, w, 4, 4, 1, 1),
                         ("RESIZE_NEAREST_NEIGHBOR", h * 2, w * 2, 4, 4, 0, 2),
                         ("RESIZE_NEAREST_NEIGHBOR", height, width, 4, 4, 0, 2),
                         ("CONV_2D", height, width, 4, 3, 1, 1)]
    if architecture != BN_ARCHITECTURE:
        raise ValueError("Unsupported deployment architecture")
    h, w = height // 4, width // 4
    layers = [("CONV_2D", height, width, 3, 8, 3, 1),
              ("CONV_2D", height // 2, width // 2, 8, 16, 3, 2),
              ("CONV_2D", h, w, 16, 32, 3, 2)]
    for _ in range(5):
        layers.extend([("CONV_2D", h, w, 32, 32, 3, 1),
                       ("CONV_2D", h, w, 32, 32, 3, 1),
                       ("ADD", h, w, 32, 32, 0, 1)])
    return layers + [("RESIZE_NEAREST_NEIGHBOR", height // 2, width // 2, 32, 32, 0, 2),
                     ("CONV_2D", height // 2, width // 2, 32, 16, 3, 1),
                     ("RESIZE_NEAREST_NEIGHBOR", height, width, 16, 16, 0, 2),
                     ("CONV_2D", height, width, 16, 8, 3, 1),
                     ("CONV_2D", height, width, 8, 3, 3, 1)]


def compute_budget(architecture, size):
    height, width = dimensions(size)
    layers = []
    for name, h, w, ci, co, kernel, stride in expected_layers(architecture, height, width):
        weights = kernel * kernel * co * (1 if name == "DEPTHWISE_CONV_2D" else ci)
        macs = h * w * weights
        layers.append({"name": name, "output_shape": [1, h, w, co], "input_channels": ci,
                       "kernel": kernel, "stride": stride, "weight_bytes": weights, "macs": macs,
                       "bias_bytes": 4 * co if kernel else 0, "output_activation_bytes": h * w * co})
    total = sum(layer["macs"] for layer in layers)
    targets = []
    for h, w, fps in ((480, 640, 15), (480, 640, 30), (480, 640, 60), (720, 1280, 60)):
        macs = total * h * w // (height * width)
        targets.append({"height": h, "width": w, "fps": fps, "macs_per_frame": macs,
                        "required_gmac_per_second": macs * fps / 1e9})
    return {"macs_per_frame": total, "weight_bytes": sum(x["weight_bytes"] for x in layers),
            "bias_bytes": sum(layer["bias_bytes"] for layer in layers),
            "largest_activation_bytes": max([height * width * 3] +
                                            [layer["output_activation_bytes"] for layer in layers]),
            "io_tensor_bytes": {"input": height * width * 3, "output": height * width * 3},
            "layers": layers, "targets": targets,
            "memory_note": "Logical INT8 tensor sizes and INT32 biases only, not an arena, peak RAM, "
                           "FlatBuffer size or FPGA resource estimate; excludes quantization tables and buffering.",
            "note": "Logical convolution MACs only; excludes resize, quantization, DMA and dispatch. Not measured FPS."}


def metrics(reference, actual):
    import numpy as np

    if reference.shape != actual.shape:
        raise ValueError(f"Parity shape mismatch: {reference.shape} vs {actual.shape}")
    if not np.isfinite(reference).all() or not np.isfinite(actual).all():
        raise ValueError("Non-finite inference output")
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    mse = float(np.mean(delta ** 2))
    return {"max_abs": float(np.max(np.abs(delta))), "mae": float(np.mean(np.abs(delta))),
            "rmse": math.sqrt(mse), "psnr_255_db": 10 * math.log10(255 ** 2 / mse) if mse else None}


def parity_metrics(reference, actual, *, atol=STRICT_PARITY_ATOL):
    import numpy as np

    result = metrics(reference, actual)
    result.update(atol=atol, rtol=STRICT_PARITY_RTOL)
    close = np.isclose(reference, actual, atol=atol, rtol=STRICT_PARITY_RTOL)
    result["passed"] = bool(np.all(close))
    result["failed_elements"] = int(np.count_nonzero(~close))
    if not result["passed"]:
        delta = np.abs(reference.astype(np.float64) - actual.astype(np.float64))
        allowed = atol + STRICT_PARITY_RTOL * np.abs(actual.astype(np.float64))
        ratios = np.where(close, -1, delta / allowed)
        index = np.unravel_index(int(np.argmax(ratios)), ratios.shape)
        result["worst_failure"] = {
            "index": [int(value) for value in index],
            "reference": float(reference[index]), "actual": float(actual[index]),
            "abs_error": float(delta[index]), "allowed_error": float(allowed[index]),
            "tolerance_ratio": float(ratios[index]),
        }
    return result


def require_parity(reference, actual, name, report):
    result = parity_metrics(reference, actual)
    report["parity"][name] = result
    if not result["passed"]:
        raise ValueError(f"{name} parity failed: {result}")


def diagnose_bn_folding(net, fused, inputs, reference, actual):
    import torch
    from neural_style.transformer_net import ConvLayer

    def layer_outputs(model):
        outputs, handles = {}, []
        for name, module in model.named_modules():
            if isinstance(module, ConvLayer):
                def capture_output(module, arguments, output, layer_name=name):
                    outputs[layer_name] = output.detach().cpu().numpy().copy()
                handles.append(module.register_forward_hook(capture_output))
        try:
            model(inputs)
        finally:
            for handle in handles:
                handle.remove()
        return outputs

    with torch.inference_mode():
        reference64_net = copy.deepcopy(net).double()
        fused64 = reference64_net.fused()
        fused64_to32 = copy.deepcopy(fused64).float()
        fused32_to64 = copy.deepcopy(fused).double()
        inputs64 = inputs.double()
        reference64 = reference64_net(inputs64).cpu().numpy()
        folded64 = fused64(inputs64).cpu().numpy()
        folded64_to32 = fused64_to32(inputs).cpu().numpy()
        folded32_to64 = fused32_to64(inputs64).cpu().numpy()
        comparisons = {
            "unfused_fp64_vs_fused_fp64": parity_metrics(reference64, folded64),
            "unfused_fp64_vs_unfused_fp32": parity_metrics(reference64, reference),
            "unfused_fp64_vs_fused_fp32": parity_metrics(reference64, actual),
            "unfused_fp64_vs_fused_fp32_to_fp64": parity_metrics(reference64, folded32_to64),
            "unfused_fp32_vs_fused_fp64_to_fp32": parity_metrics(reference, folded64_to32),
            "unfused_fp64_vs_fused_fp64_to_fp32": parity_metrics(reference64, folded64_to32),
        }
        original_layers, folded_layers = layer_outputs(net), layer_outputs(fused)
        layers = []
        for name, layer in net.named_modules():
            if not isinstance(layer, ConvLayer):
                continue
            entry = {"name": name, "parity": parity_metrics(original_layers[name], folded_layers[name])}
            if isinstance(layer.bn, torch.nn.BatchNorm2d):
                batchnorm = layer.bn
                scale = batchnorm.weight.double() * torch.rsqrt(batchnorm.running_var.double() + batchnorm.eps)
                entry["batchnorm"] = {
                    "eps": batchnorm.eps,
                    "running_var_min": float(batchnorm.running_var.min()),
                    "running_var_max": float(batchnorm.running_var.max()),
                    "running_mean_abs_max": float(batchnorm.running_mean.abs().max()),
                    "scale_abs_max": float(scale.abs().max()),
                }
            layers.append(entry)
    first_failed_layer = next((layer["name"] for layer in layers if not layer["parity"]["passed"]), None)
    return {
        "torch_version": str(torch.__version__),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled), "threads": torch.get_num_threads(),
        "input_shape": list(inputs.shape), "comparisons": comparisons,
        "first_failed_layer": first_failed_layer, "layers": layers,
        "scope": "Read-only diagnostics for the failing probe. No weights, tolerances or conversion gates are changed.",
        "layer_note": "Layer outputs include upstream errors; first_failed_layer is localization, not proof of a faulty layer.",
    }


def check_folding_parity(net, fused, probes, report, experimental_atol=None):
    """Return original FP32 fused outputs; experimental budgets never replace strict results."""
    import torch

    references = []
    experiment = None
    if experimental_atol is not None:
        if not math.isfinite(experimental_atol) or experimental_atol <= STRICT_PARITY_ATOL:
            raise ValueError("Experimental folding tolerance must be finite and greater than 0.01")
        experiment = report["folding_experiment"] = {
            "atol": experimental_atol, "rtol": STRICT_PARITY_RTOL, "passed": False,
            "strict_failures": [], "failures": [], "probes": [],
            "scope": "Experimental FP32 folding budget in unclipped RGB units only. "
                     "Strict FP32 results remain in parity; FP64 equivalence and downstream tolerances stay strict. "
                     "Export weights are unchanged; artifacts are not strict-acceptance candidates.",
        }
        print(f"[experimental] folding atol={experimental_atol}, rtol={STRICT_PARITY_RTOL}; "
              "strict results are retained; completed conversion exits 3", file=sys.stderr, flush=True)
    with torch.inference_mode():
        if experiment is not None:
            reference64_net = copy.deepcopy(net).double()
            fused64 = reference64_net.fused()
        for index, probe in enumerate(probes):
            inputs = torch.from_numpy(probe)
            reference = net(inputs).numpy()
            actual = fused(inputs).numpy()
            if actual.shape != probe.shape:
                raise ValueError(f"Model output must preserve RGB image shape, got {actual.shape}")
            name = f"pytorch_unfused_vs_fused_{index}"
            strict = report["parity"][name] = parity_metrics(reference, actual)
            if experiment is not None:
                if not strict["passed"]:
                    experiment["strict_failures"].append(name)
                budget = parity_metrics(reference, actual, atol=experimental_atol)
                inputs64 = inputs.double()
                equivalence = parity_metrics(reference64_net(inputs64).numpy(), fused64(inputs64).numpy())
                experiment["probes"].append({
                    "probe_index": index, "strict_parity_key": name,
                    "fp32_budget": budget, "fp64_equivalence": equivalence,
                })
                if not budget["passed"]:
                    experiment["failures"].append(f"{name}: experimental FP32 folding budget exceeded")
                if not equivalence["passed"]:
                    experiment["failures"].append(f"{name}: strict FP64 folding equivalence failed")
                print(f"[experimental] probe={index} strict={strict['passed']} "
                      f"fp32_budget={budget['passed']} fp64_equivalence={equivalence['passed']}",
                      file=sys.stderr, flush=True)
            elif not strict["passed"]:
                try:
                    diagnostics = diagnose_bn_folding(net, fused, inputs, reference, actual)
                    diagnostics["probe_index"] = index
                    report["folding_diagnostics"] = diagnostics
                    for comparison_name, comparison in diagnostics["comparisons"].items():
                        print(f"[folding] {comparison_name}: passed={comparison['passed']} "
                              f"max_abs={comparison['max_abs']:.8g} failed_elements={comparison['failed_elements']}",
                              file=sys.stderr, flush=True)
                    print(f"[folding] first_failed_layer={diagnostics['first_failed_layer']}",
                          file=sys.stderr, flush=True)
                except Exception as diagnostic_error:
                    report["folding_diagnostics"] = {
                        "probe_index": index, "error": f"{type(diagnostic_error).__name__}: {diagnostic_error}",
                        "traceback": traceback.format_exc(),
                    }
                raise ValueError(f"{name} parity failed: {strict}")
            references.append(actual)
    if experiment is not None:
        experiment["strict_folding_passed"] = not experiment["strict_failures"]
        experiment["passed"] = bool(probes) and not experiment["failures"]
        if not experiment["passed"]:
            raise ValueError("Experimental folding checks failed: " +
                             "; ".join(experiment["failures"] or ["No input probes"]))
    return references


def build_fused_fp64_references(net, fused, probes, report):
    """Evaluate copies of the actual FP32 export parameters in FP64, without re-fusing."""
    import torch

    details = report["fp64_reference"] = {
        "source": "exported_fp32_parameters_evaluated_in_fp64",
        "validated": False, "probes": [], "failures": [], "parity": {}, "strict_fp32_failures": [],
        "scope": "Explicit experimental reference policy, not original FP32 acceptance. "
                 "Export parameters are only upcast on a copy, never re-fused or changed. "
                 "Parameter equivalence and downstream FP64-reference comparisons use strict tolerances.",
    }
    references = []
    with torch.inference_mode():
        original64 = copy.deepcopy(net).double()
        exported64 = copy.deepcopy(fused).double()
        for index, probe in enumerate(probes):
            inputs = torch.from_numpy(probe).double()
            reference = original64(inputs).numpy()
            actual = exported64(inputs).numpy()
            if actual.shape != probe.shape:
                raise ValueError(f"FP64 reference must preserve RGB image shape, got {actual.shape}")
            result = parity_metrics(reference, actual)
            details["probes"].append({"probe_index": index, "unfused_vs_exported_weights": result})
            if not result["passed"]:
                details["failures"].append(f"probe {index}: FP32 export parameters fail strict FP64 equivalence")
            print(f"[reference] probe={index} export_parameters={result['passed']} "
                  f"max_abs={result['max_abs']:.8g} failed_elements={result['failed_elements']}",
                  file=sys.stderr, flush=True)
            references.append(actual)
    if len(probes) != 3:
        details["failures"].append("FP64 reference requires all three input probes")
    details["validated"] = not details["failures"]
    if not details["validated"]:
        raise ValueError("FP64 reference validation failed: " + "; ".join(details["failures"]))
    return references


def require_export_parity(reference, actual, name, report, fp64_reference=None):
    """Retain FP32 results; only an explicit validated FP64 reference selects the experimental gate."""
    if fp64_reference is None:
        require_parity(reference, actual, name, report)
        return
    details = report.get("fp64_reference", {})
    if details.get("validated") is not True or len(details.get("probes", [])) != 3:
        raise ValueError("A validated three-probe FP64 reference is required")
    original = report["parity"][name] = parity_metrics(reference, actual)
    if not original["passed"]:
        details["strict_fp32_failures"].append(name)
    result = details["parity"][name] = parity_metrics(fp64_reference, actual)
    if not original["passed"]:
        index = tuple(original["worst_failure"]["index"])
        result["reference_at_fp32_worst_failure"] = float(fp64_reference[index])
    print(f"[reference] {name}: fp32={original['passed']} fp64_reference={result['passed']} "
          f"max_abs={result['max_abs']:.8g} failed_elements={result['failed_elements']}",
          file=sys.stderr, flush=True)
    if not result["passed"]:
        raise ValueError(f"{name} FP64-reference parity failed: {result}")


def signature_layout(spec, size):
    height, width = dimensions(size)
    shape = spec.shape.as_list()
    if spec.dtype.name != "float32":
        raise ValueError(f"Expected float32 SavedModel signature, got {spec}")
    if shape == [1, height, width, 3]:
        return "NHWC"
    if shape == [1, 3, height, width]:
        return "NCHW"
    raise ValueError(f"SavedModel signature must be static batch 1 RGB: {spec}")


def validate_graph(tensors, ops, profile, blockers, subgraph):
    """Validate topology as well as counts, including linear heads and DW axes."""
    prefix = f"subgraph {subgraph}"
    if len(ops) != len(profile):
        blockers.append(f"{prefix}: expected {len(profile)} operators, got {len(ops)}")
        return
    current = ops[0]["inputs"][0]
    skip = None
    residual_starts = {index - 2 for index, row in enumerate(profile) if row[0] == "ADD"}
    for index, (op, expected) in enumerate(zip(ops, profile)):
        name, h, w, ci, co, kernel, stride = expected
        label = f"{prefix} op {index}"
        if op["name"] != name or len(op["outputs"]) != 1:
            blockers.append(f"{label}: expected {name} with one output")
            continue
        inputs, output = op["inputs"], op["outputs"][0]
        required_inputs = 3 if kernel else 2
        if len(inputs) != required_inputs or any(i < 0 or i >= len(tensors) for i in inputs + [output]):
            blockers.append(f"{label}: invalid tensor references or input count")
            continue
        if index in residual_starts:
            skip = current
        if name == "ADD":
            if skip is None or sorted(inputs) != sorted([current, skip]):
                blockers.append(f"{label}: residual inputs do not match the block")
        elif inputs[0] != current:
            blockers.append(f"{label}: input does not follow the declared pipeline")
        src, dst = tensors[inputs[0]], tensors[output]
        in_h = h * stride if kernel else h // stride if name == "RESIZE_NEAREST_NEIGHBOR" else h
        in_w = w * stride if kernel else w // stride if name == "RESIZE_NEAREST_NEIGHBOR" else w
        if src["shape"] != [1, in_h, in_w, ci] or dst["shape"] != [1, h, w, co]:
            blockers.append(f"{label}: unexpected feature shapes/channels")
        for tensor in [tensors[i] for i in inputs if i == current or name == "ADD"] + [dst]:
            if tensor["dtype"] != "INT8" or len(tensor["scales"]) != 1:
                blockers.append(f"{label}: activations must use per-tensor INT8 quantization")
        opts = op["options"]
        activation = "NONE"
        if kernel:
            linear = index == len(profile) - 1 or profile[index + 1][0] == "ADD"
            activation = "NONE" if linear else "RELU"
            if opts.get("StrideH") != stride or opts.get("StrideW") != stride:
                blockers.append(f"{label}: unexpected stride")
            weight, bias = tensors[inputs[1]], tensors[inputs[2]]
            dw = name == "DEPTHWISE_CONV_2D"
            shape = [1, kernel, kernel, co] if dw else [co, kernel, kernel, ci]
            if (weight["shape"] != shape or not weight["constant"] or
                    weight["dtype"] != "INT8" or weight["quantized_dimension"] != (3 if dw else 0) or
                    len(weight["scales"]) != co or weight["zero_points"] != [0] * co):
                blockers.append(f"{label}: requires constant symmetric per-output-channel INT8 weights")
            if (bias["shape"] != [co] or bias["dtype"] != "INT32" or not bias["constant"] or
                    bias["quantized_dimension"] != 0 or len(bias["scales"]) != co or
                    bias["zero_points"] != [0] * co):
                blockers.append(f"{label}: requires constant per-channel INT32 bias")
            elif len(src["scales"]) == 1 and len(weight["scales"]) == co:
                if any(not math.isclose(b, src["scales"][0] * s, rel_tol=1e-5, abs_tol=1e-12)
                       for b, s in zip(bias["scales"], weight["scales"])):
                    blockers.append(f"{label}: bias scales differ from input_scale * weight_scale")
        elif name == "RESIZE_NEAREST_NEIGHBOR":
            shape_tensor = tensors[inputs[1]]
            if shape_tensor["shape"] != [2] or not shape_tensor["constant"]:
                blockers.append(f"{label}: requires constant two-element resize shape")
        if opts.get("FusedActivationFunction", "NONE") != activation:
            blockers.append(f"{label}: unexpected activation; RGB head and residual second conv must be linear")
        current = output


def audit_tflite(data, size, architecture=BN_ARCHITECTURE):
    """Read every subgraph using public generated FlatBuffer bindings, no delegates."""
    import tflite

    height, width = dimensions(size)
    profile = expected_layers(architecture, height, width)
    model = tflite.Model.GetRootAsModel(data, 0)
    enum_names = lambda cls: {v: k for k, v in vars(cls).items() if isinstance(v, int)}
    op_names = enum_names(tflite.BuiltinOperator)
    type_names = enum_names(tflite.TensorType)
    activations = enum_names(tflite.ActivationFunctionType)
    paddings = enum_names(tflite.Padding)
    blockers, graphs, largest = [], [], None
    allowed = {row[0] for row in profile}
    byte_sizes = {"INT8": 1, "INT32": 4, "INT64": 8, "FLOAT32": 4, "FLOAT16": 2, "UINT8": 1}
    if model.SubgraphsLength() != 1:
        blockers.append("Expected exactly one subgraph")
    for g in range(model.SubgraphsLength()):
        graph = model.Subgraphs(g)
        tensors, ops = [], []
        bias_or_shape = set()
        data_tensors = set()
        for j in range(graph.OperatorsLength()):
            op = graph.Operators(j)
            code = model.OperatorCodes(op.OpcodeIndex())
            name = op_names.get(max(code.BuiltinCode(), code.DeprecatedBuiltinCode()), "UNKNOWN")
            inputs = [int(op.Inputs(k)) for k in range(op.InputsLength())]
            outputs = [int(op.Outputs(k)) for k in range(op.OutputsLength())]
            options = {}
            table = op.BuiltinOptions()
            option_class = {"CONV_2D": "Conv2DOptions", "DEPTHWISE_CONV_2D": "DepthwiseConv2DOptions", "ADD": "AddOptions",
                            "RESIZE_NEAREST_NEIGHBOR": "ResizeNearestNeighborOptions"}.get(name)
            if table is not None and option_class:
                obj = getattr(tflite, option_class)()
                obj.Init(table.Bytes, table.Pos)
                for field in ("Padding", "StrideH", "StrideW", "DilationHFactor", "DilationWFactor",
                               "FusedActivationFunction", "AlignCorners", "HalfPixelCenters", "PotScaleInt16", "DepthMultiplier"):
                    if hasattr(obj, field):
                        value = getattr(obj, field)()
                        if field == "Padding":
                            value = paddings.get(value, value)
                        if field == "FusedActivationFunction":
                            value = activations.get(value, value)
                        options[field] = value
            entry = {"index": j, "name": name, "version": int(code.Version()),
                     "inputs": inputs, "outputs": outputs, "options": options}
            ops.append(entry)
            if name not in allowed:
                blockers.append(f"subgraph {g} op {j}: unsupported {name}; ReLU must be fused")
            if options.get("FusedActivationFunction", "NONE") not in ("NONE", "RELU"):
                blockers.append(f"subgraph {g} op {j}: unsupported fused activation")
            if name in ("CONV_2D", "DEPTHWISE_CONV_2D") and len(inputs) == 3:
                bias_or_shape.add(inputs[2])
                data_tensors.update(inputs[:2])
                weight = graph.Tensors(inputs[1])
                kernel = [weight.Shape(k) for k in range(weight.ShapeLength())]
                valid_kernel = [1, 1] if name == "CONV_2D" and architecture == DW_ARCHITECTURE else [3, 3]
                if len(kernel) != 4 or kernel[1:3] != valid_kernel:
                    blockers.append(f"subgraph {g} op {j}: unexpected convolution kernel")
                if name == "DEPTHWISE_CONV_2D" and options.get("DepthMultiplier") != 1:
                    blockers.append(f"subgraph {g} op {j}: depth_multiplier must be one")
                # A 1x1 stride-one filter has no boundary padding in either mode.
                valid_padding = options.get("Padding") == "SAME" or (
                    kernel[1:3] == [1, 1] and options.get("StrideH") == 1 and options.get("Padding") == "VALID")
                if (not valid_padding or
                        options.get("StrideH") not in (1, 2) or
                        options.get("StrideH") != options.get("StrideW") or
                        options.get("DilationHFactor") != 1 or options.get("DilationWFactor") != 1):
                    blockers.append(f"subgraph {g} op {j}: requires SAME (or equivalent 1x1 VALID), stride 1/2, dilation 1")
            elif name == "RESIZE_NEAREST_NEIGHBOR" and len(inputs) == 2:
                bias_or_shape.add(inputs[1])
                data_tensors.add(inputs[0])
            else:
                data_tensors.update(i for i in inputs if i >= 0)
            data_tensors.update(i for i in outputs if i >= 0)
        for j in range(graph.TensorsLength()):
            tensor = graph.Tensors(j)
            shape = [int(tensor.Shape(k)) for k in range(tensor.ShapeLength())]
            signature = [int(tensor.ShapeSignature(k)) for k in range(tensor.ShapeSignatureLength())]
            dtype = type_names.get(tensor.Type(), f"UNKNOWN_{tensor.Type()}")
            q = tensor.Quantization()
            scales = [float(q.Scale(k)) for k in range(q.ScaleLength())] if q else []
            zps = [int(q.ZeroPoint(k)) for k in range(q.ZeroPointLength())] if q else []
            nbytes = math.prod(shape) * byte_sizes.get(dtype, 0)
            entry = {"index": j, "name": (tensor.Name() or b"").decode("utf-8", errors="replace"),
                     "shape": shape, "shape_signature": signature, "dtype": dtype,
                     "scales": scales, "zero_points": zps,
                     "quantized_dimension": int(q.QuantizedDimension()) if q else None,
                     "bytes": nbytes, "constant": bool(model.Buffers(tensor.Buffer()).DataLength())}
            tensors.append(entry)
            if largest is None or nbytes > largest["bytes"]:
                largest = {"subgraph": g, **entry}
            if any(d <= 0 for d in shape) or any(d <= 0 for d in signature):
                blockers.append(f"subgraph {g} tensor {j}: non-static shape")
            if dtype != "INT8" and not (dtype == "INT32" and j in bias_or_shape and j not in data_tensors):
                blockers.append(f"subgraph {g} tensor {j}: forbidden dtype {dtype} (only int8 data/int32 bias or shape)")
            if dtype == "INT8" or (dtype == "INT32" and j in bias_or_shape and scales):
                if not scales or len(scales) != len(zps) or not all(math.isfinite(s) and s > 0 for s in scales):
                    blockers.append(f"subgraph {g} tensor {j}: invalid quantization parameters")
            if dtype == "INT8" and any(z < -128 or z > 127 for z in zps):
                blockers.append(f"subgraph {g} tensor {j}: zero point outside int8 range")
        inputs = [int(graph.Inputs(k)) for k in range(graph.InputsLength())]
        outputs = [int(graph.Outputs(k)) for k in range(graph.OutputsLength())]
        if len(inputs) != 1 or len(outputs) != 1:
            blockers.append(f"subgraph {g}: expected one input and one output")
        for j in inputs + outputs:
            tensor = tensors[j]
            if tensor["shape"] != [1, height, width, 3] or tensor["dtype"] != "INT8" or len(tensor["scales"]) != 1:
                blockers.append(f"subgraph {g} IO tensor {j}: requires static NHWC [1,{height},{width},3], per-tensor int8")
        if ops and (ops[0]["inputs"][:1] != inputs or ops[-1]["outputs"] != outputs):
            blockers.append(f"subgraph {g}: graph IO does not match the declared pipeline endpoints")
        for op in ops:
            if op["name"] == "RESIZE_NEAREST_NEIGHBOR":
                if op["options"].get("AlignCorners", False) or op["options"].get("HalfPixelCenters", False):
                    blockers.append(f"subgraph {g} op {op['index']}: both resize coordinate flags must be false")
                src, dst = tensors[op["inputs"][0]], tensors[op["outputs"][0]]
                if src["scales"] != dst["scales"] or src["zero_points"] != dst["zero_points"]:
                    blockers.append(f"subgraph {g} op {op['index']}: resize quantization must be identical")
                s, d = src["shape"], dst["shape"]
                if len(s) != 4 or d != [s[0], s[1] * 2, s[2] * 2, s[3]]:
                    blockers.append(f"subgraph {g} op {op['index']}: resize must be nearest 2x NHWC")
        validate_graph(tensors, ops, profile, blockers, g)
        graphs.append({"index": g, "inputs": inputs, "outputs": outputs, "operators": ops, "tensors": tensors,
                       "operator_counts": {name: sum(op["name"] == name for op in ops)
                                           for name in sorted({op["name"] for op in ops})}})
    return {"blockers": blockers, "passed": not blockers, "model_bytes": len(data),
            "largest_tensor": largest, "memory_note": "Largest logical tensor only; not an arena or peak RAM estimate.",
            "subgraphs": graphs}


def canonicalize_convolutions(model):
    """Fold constant zero Pad into Conv and prove the static SAME_UPPER bounds."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper
    from onnxsim import simplify

    model, checked = simplify(model)
    if not checked:
        raise ValueError("ONNX simplification validation failed")
    constants = {v.name: numpy_helper.to_array(v) for v in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        pad = producers.get(node.input[0])
        if pad is None or pad.op_type != "Pad":
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in pad.attribute}
        values = constants.get(pad.input[1]) if len(pad.input) > 1 else None
        value = constants.get(pad.input[2]) if len(pad.input) > 2 and pad.input[2] else np.array(0)
        if (attrs.get("mode", b"constant") != b"constant" or values is None or
                value is None or not np.all(value == 0) or values.shape != (8,) or
                np.any(values[[0, 1, 4, 5]] != 0) or np.any(values < 0)):
            raise ValueError("Cannot fold non-constant/nonzero/spatially invalid padding")
        conv_attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        old = conv_attrs.get("pads", [0, 0, 0, 0])
        combined = [int(x + y) for x, y in zip(old, values[[2, 3, 6, 7]])]
        kept = [a for a in node.attribute if a.name != "pads"]
        del node.attribute[:]
        node.attribute.extend(kept + [helper.make_attribute("pads", combined)])
        node.input[0] = pad.input[0]
    used = {name for node in model.graph.node for name in node.input}
    used.update(value.name for value in model.graph.output)
    kept = [node for node in model.graph.node if node.op_type != "Pad" or any(name in used for name in node.output)]
    del model.graph.node[:]
    model.graph.node.extend(kept)
    model = onnx.shape_inference.infer_shapes(model)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in list(model.graph.input) + list(model.graph.value_info)}
    for node in model.graph.node:
        if node.op_type != "Conv":
            continue
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        shape = shapes.get(node.input[0], [])
        kernel = constants[node.input[1]].shape[2:]
        strides = attrs.get("strides", [1, 1])
        weight = constants[node.input[1]]
        group = attrs.get("group", 1)
        depthwise = len(shape) == 4 and group == shape[1] and weight.shape[0] == shape[1] and weight.shape[1] == 1
        if (len(shape) != 4 or min(shape) <= 0 or tuple(kernel) not in ((3, 3), (1, 1)) or
                attrs.get("dilations", [1, 1]) != [1, 1] or (group != 1 and not depthwise)):
            raise ValueError("Expected static dense or multiplier-one depthwise convolution, dilation one")
        total = [max(((n + s - 1) // s - 1) * s + k - n, 0)
                 for n, s, k in zip(shape[2:], strides, kernel)]
        expected = [p // 2 for p in total] + [p - p // 2 for p in total]
        if attrs.get("pads", [0, 0, 0, 0]) != expected:
            raise ValueError("Convolution padding differs from SAME_UPPER")
        kept = [a for a in node.attribute if a.name not in ("pads", "auto_pad")]
        del node.attribute[:]
        node.attribute.extend(kept + [helper.make_attribute("auto_pad", "SAME_UPPER")])
    onnx.checker.check_model(model)
    return model


def convert(a, report):
    record_stage(report, "import_numpy")
    import numpy as np
    record_stage(report, "import_torch")
    import torch
    record_stage(report, "import_onnx")
    import onnx
    record_stage(report, "import_onnxruntime")
    import onnxruntime as ort
    record_stage(report, "import_tensorflow")
    import tensorflow as tf
    record_stage(report, "import_model")
    from neural_style import transformer_net

    if not hasattr(transformer_net, "load_checkpoint"):
        raise ValueError("Model module must expose load_checkpoint(path) and net.fused(); legacy checkpoints are unsupported")
    record_stage(report, "load_checkpoint")
    net, metadata = transformer_net.load_checkpoint(a.model)
    record_stage(report, "validate_checkpoint")
    checkpoint_shape = transformer_net.checkpoint_dimensions(metadata)
    architecture = metadata["architecture"]
    if architecture == BN_ARCHITECTURE:
        width = float(metadata["width"])
        if not math.isclose(width, 0.25, rel_tol=0, abs_tol=1e-9):
            raise ValueError(f"Legacy Ti60 conversion requires width 0.25, got {width}")
        if a.width is not None and not math.isclose(a.width, width, rel_tol=0, abs_tol=1e-9):
            raise ValueError("--width differs from checkpoint metadata")
    elif a.width is not None:
        raise ValueError("--width is only supported for the legacy BN architecture")
    a.size = dimensions((a.height, a.input_width) if a.height is not None else
                        a.size if a.size is not None else checkpoint_shape)
    height, image_width = a.size
    record_stage(report, "discover_calibration")
    files = sorted(p for p in Path(a.calib).rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:a.calib_count]
    if not files:
        raise ValueError(f"No calibration jpg/jpeg/png images under {a.calib}")
    record_stage(report, "prepare_model_report")
    report.update(metadata=metadata, size=list(a.size), input_shape=[1, height, image_width, 3],
                  compute=compute_budget(architecture, a.size),
                  calibration={"count": len(files), "files": [str(p.resolve()) for p in files],
                  "preprocessing": "RGB PIL bilinear aspect-fill Resize, CenterCrop, float32 RGB 0..255"})
    report["target_contract"].update(architecture=architecture,
        input_shape=[1, height, image_width, 3],
        operator_counts={name: sum(row[0] == name for row in expected_layers(architecture, *a.size))
                         for name in sorted({row[0] for row in expected_layers(architecture, *a.size)})})
    record_stage(report, "fuse_batchnorm")
    if any(m.training for m in net.modules()) or any(t.device.type != "cpu" for t in list(net.parameters()) + list(net.buffers())):
        raise ValueError("load_checkpoint must return an eval CPU model")
    fused = net.fused()
    if fused is net or any(m.training for m in fused.modules()):
        raise ValueError("fused() must return a separate eval model")
    if any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in fused.modules()):
        raise ValueError("fused() left BatchNorm modules in the export model")
    if any(t.device.type != "cpu" for t in list(fused.parameters()) + list(fused.buffers())):
        raise ValueError("fused() must return a CPU model")
    original_storage = {t.untyped_storage().data_ptr() for t in list(net.parameters()) + list(net.buffers()) if t.numel()}
    if any(t.numel() and t.untyped_storage().data_ptr() in original_storage
           for t in list(fused.parameters()) + list(fused.buffers())):
        raise ValueError("fused() must deepcopy the model, not share parameter/buffer storage")
    convs = [m for m in fused.modules() if isinstance(m, torch.nn.Conv2d)]
    expected_convs = [row for row in expected_layers(architecture, *a.size) if row[5]]
    if len(convs) != len(expected_convs):
        raise ValueError("Unexpected convolution count in checkpoint")
    for conv, (name, _, _, ci, co, kernel, stride) in zip(convs, expected_convs):
        if (conv.in_channels != ci or conv.out_channels != co or conv.kernel_size != (kernel, kernel) or
                conv.groups != (ci if name == "DEPTHWISE_CONV_2D" else 1) or
                conv.stride != (stride, stride) or conv.dilation != (1, 1) or conv.padding_mode != "zeros"):
            raise ValueError("Checkpoint convolution differs from deployment profile")
    record_stage(report, "prepare_inputs")
    image = preprocess(a.content or files[0], a.size)
    rng = np.random.default_rng(0)
    probes = [image.transpose(0, 3, 1, 2).copy(),
               rng.uniform(0, 255, (1, 3, height, image_width)).astype(np.float32),
               np.zeros((1, 3, height, image_width), dtype=np.float32)]
    record_stage(report, "pytorch_parity")
    references = check_folding_parity(net, fused, probes, report, a.experimental_folding_atol)
    fp64_references = None
    if a.experimental_fused_fp64_reference:
        record_stage(report, "fp64_reference_validation")
        fp64_references = build_fused_fp64_references(net, fused, probes, report)
    with torch.inference_mode():
        record_stage(report, "onnx_export")
        torch.onnx.export(fused, torch.from_numpy(probes[0]), a.onnx, opset_version=13,
                          dynamo=False, input_names=["input"], output_names=["output"],
                          dynamic_axes=None, do_constant_folding=True)
    record_stage(report, "onnx_canonicalization")
    exported = canonicalize_convolutions(onnx.load(a.onnx))
    onnx.checker.check_model(exported)
    onnx.save(exported, a.onnx)
    record_stage(report, "onnx_parity")
    session = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    for i, x in enumerate(probes):
        out = session.run(None, {session.get_inputs()[0].name: x})[0]
        require_export_parity(references[i], out, f"pytorch_fused_vs_onnx_{i}", report,
                              fp64_references[i] if fp64_references is not None else None)
    report["artifacts"]["onnx"] = str(Path(a.onnx).resolve())
    record_stage(report, "onnx_to_saved_model")
    saved = Path(tempfile.mkdtemp(prefix="run-", dir=a.saved_model))
    report["artifacts"]["saved_model"] = str(saved.resolve())
    subprocess.run(["onnx2tf", "-i", str(Path(a.onnx).resolve()), "-o", str(saved.resolve()), "-b", "1"], check=True)
    record_stage(report, "saved_model_parity")
    loaded = tf.saved_model.load(str(saved))
    if "serving_default" not in loaded.signatures:
        raise ValueError("SavedModel lacks serving_default signature")
    signature = loaded.signatures["serving_default"]
    positional, keyword = signature.structured_input_signature
    if positional or len(keyword) != 1 or not isinstance(signature.structured_outputs, dict) or len(signature.structured_outputs) != 1:
        raise ValueError("SavedModel requires one named input and one named output")
    input_name, input_spec = next(iter(keyword.items()))
    output_name, output_spec = next(iter(signature.structured_outputs.items()))
    layout = signature_layout(input_spec, a.size)
    output_layout = signature_layout(output_spec, a.size)
    report["saved_model_signature"] = {"key": "serving_default", "input": input_name, "input_layout": layout,
                                        "input_shape": input_spec.shape.as_list(), "output": output_name,
                                        "output_layout": output_layout, "output_shape": output_spec.shape.as_list()}
    for i, x in enumerate(probes):
        sx = x.transpose(0, 2, 3, 1) if layout == "NHWC" else x
        out = signature(**{input_name: tf.convert_to_tensor(sx)})[output_name].numpy()
        if output_layout == "NHWC":
            out = out.transpose(0, 3, 1, 2)
        require_export_parity(references[i], out, f"pytorch_fused_vs_saved_model_{i}", report,
                              fp64_references[i] if fp64_references is not None else None)

    def representative_dataset():
        for path in files:
            pixels = preprocess(path, a.size)
            yield {input_name: pixels if layout == "NHWC" else pixels.transpose(0, 3, 1, 2).copy()}

    record_stage(report, "tflite_quantization")
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved), signature_keys=["serving_default"])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    data = converter.convert()
    Path(a.int8_tflite).write_bytes(data)
    report["artifacts"]["int8_tflite"] = str(Path(a.int8_tflite).resolve())
    record_stage(report, "tflite_audit")
    report["audit"] = audit_tflite(data, a.size, architecture)
    report["blockers"].extend(report["audit"]["blockers"])
    if report["blockers"]:
        return
    record_stage(report, "int8_verification")
    interpreter = tf.lite.Interpreter(model_content=data, experimental_delegates=[],
        experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interpreter.allocate_tensors()
    inp, outp = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    scale_in, zp_in = inp["quantization"]
    scale_out, zp_out = outp["quantization"]
    if scale_in <= 0 or scale_out <= 0:
        raise ValueError("Invalid per-tensor IO quantization")
    raw = np.rint(image / scale_in + zp_in)
    quantized = np.clip(raw, -128, 127).astype(np.int8)
    interpreter.set_tensor(inp["index"], quantized)
    interpreter.invoke()
    codes = interpreter.get_tensor(outp["index"])
    dequantized = (codes.astype(np.float32) - zp_out) * scale_out
    reference = references[0].transpose(0, 2, 3, 1)
    report["verification"] = {"image": str(Path(a.content or files[0]).resolve()),
        "float_reference": "pytorch_fused_fp32",
        "float_vs_int8_unclipped": metrics(reference, dequantized),
        "float_vs_int8_clipped_0_255": metrics(np.clip(reference, 0, 255), np.clip(dequantized, 0, 255)),
        "input_clipped_fraction": float(np.mean((raw < -128) | (raw > 127))),
        "output_at_int8_limits_fraction": float(np.mean((codes == -128) | (codes == 127))),
        "quality_note": "Measured on one image; no universal int8 quality threshold is asserted.",
        "io_formulas": {"quantize": "clip(round(real / scale + zero_point), -128, 127)",
                        "dequantize": "(int8_code - zero_point) * scale",
                        "preview": "round(clip(dequantized_RGB, 0, 255)) as uint8"},
        "input_quantization": {"scale": scale_in, "zero_point": zp_in},
        "output_quantization": {"scale": scale_out, "zero_point": zp_out}}
    from PIL import Image
    Image.fromarray(np.rint(np.clip(dequantized[0], 0, 255)).astype(np.uint8)).save(a.output_image, format="PNG")
    report["artifacts"]["preview"] = str(Path(a.output_image).resolve())


def main():
    a = parse_args()
    a.report = a.report or a.int8_tflite + ".report.json"
    a.output_image = a.output_image or a.int8_tflite + ".preview.png"
    paths = [Path(p).resolve() for p in (a.onnx, a.int8_tflite, a.report, a.output_image)]
    protected = {Path(a.model).resolve()}
    if a.content:
        protected.add(Path(a.content).resolve())
    protected.update(p.resolve() for p in Path(a.calib).rglob("*")
                     if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(set(paths)) != len(paths) or protected.intersection(paths):
        # Do not write a failure report over an input file or another artifact.
        print("Output paths must be distinct and must not overwrite input images/checkpoints", file=sys.stderr)
        return 2
    report = {"status": "failed", "target": "Ti60", "blockers": [], "parity": {}, "artifacts": {},
        "mode": ("experimental_fused_fp64_reference" if a.experimental_fused_fp64_reference else
                 "strict" if a.experimental_folding_atol is None else "experimental_folding"),
        "experimental_folding_atol": a.experimental_folding_atol,
        "experimental_fused_fp64_reference": a.experimental_fused_fp64_reference,
        "target_contract": {"batch": 1, "io_layout": "NHWC", "io_dtype": "int8",
                             "convolution": "zero SAME, dilation 1; DW multiplier 1; stride 2 pads right/bottom",
                             "upsampling": "two nearest-neighbor 2x resizes",
                             "target": "640x480 at 30 fresh frames/s; unproven"},
        "hardware_required": [
            "Nearest-neighbor 2x resize must execute in the hardware pipeline with identical input/output quantization.",
            "Pipeline boundary RGB preprocessing, scale/zero-point quantization and output dequantization/clipping need an explicit implementation and placement.",
            "Register the actual graph's Conv/Add/Resize operators (and Depthwise only when present); "
            "strictly guard hardware dispatch before any CPU fallback and verify each layer's mode.",
            "Regenerate TinyML sizing for the actual model and parallelism; confirm op versions/options, per-channel quantization, SAME padding and memory/scheduling on Ti60."],
        "hardware_proven": False,
        "scope": "Software parity and FlatBuffer compatibility screen only; no Ti60 compilation, placement or hardware execution proof."}
    try:
        record_stage(report, "prepare_outputs")
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        Path(a.saved_model).mkdir(parents=True, exist_ok=True)
        convert(a, report)
        if not report["blockers"]:
            if a.experimental_folding_atol is None:
                report["status"] = "software_checks_passed_hardware_unproven"
                record_stage(report, "complete")
            else:
                experiment = report.get("folding_experiment", {})
                if experiment.get("passed") is not True or len(experiment.get("probes", [])) != 3:
                    raise ValueError("Experimental conversion requires three passing folding probes")
                if a.experimental_fused_fp64_reference:
                    reference = report.get("fp64_reference", {})
                    if reference.get("validated") is not True or len(reference.get("probes", [])) != 3:
                        raise ValueError("Experimental conversion requires a validated three-probe FP64 reference")
                report["status"] = "experimental_conversion_completed_hardware_unproven"
                record_stage(report, "experimental_complete")
    except Exception as exc:
        report["blockers"].append(f"{type(exc).__name__}: {exc}")
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], file=sys.stderr, end="", flush=True)
    finally:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {a.report}")
    for blocker in report["blockers"]:
        print(f"BLOCKER: {blocker}", file=sys.stderr)
    if report["blockers"]:
        return 1
    if a.experimental_folding_atol is not None:
        print("EXPERIMENTAL: conversion completed, not strict acceptance; exit code 3.", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
