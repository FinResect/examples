"""Convert the metadata-bearing BN model to a candidate Ti60 int8 artifact.

Every invocation runs every stage. The old skip flags are intentionally removed.
An operator audit is a necessary condition, not proof of hardware deployment.
"""

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, help="Checkpoint with architecture/width/image_size metadata")
    p.add_argument("--width", type=float, help="Consistency check only; checkpoint width must be 0.25")
    p.add_argument("--size", type=int, help="Static square size; defaults to checkpoint image_size; multiple of 4")
    p.add_argument("--onnx", default="style.onnx")
    p.add_argument("--saved-model", default="saved_model", help="Parent for a fresh, retained run directory")
    p.add_argument("--int8-tflite", default="style_int8.tflite")
    p.add_argument("--report", help="JSON report; defaults to <int8-tflite>.report.json")
    p.add_argument("--calib", default="calib", help="Recursive RGB jpg/jpeg/png calibration directory")
    p.add_argument("--calib-count", type=int, default=100, help="Maximum calibration images (default: 100)")
    p.add_argument("--content", help="Verification image; defaults to first calibration image")
    p.add_argument("--output-image", help="Clipped verification PNG; defaults to <int8-tflite>.preview.png")
    return p.parse_args()


def preprocess(path, size):
    """PIL Resize(int, BILINEAR) + CenterCrop, then ToTensor().mul(255)."""
    import numpy as np
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        if width <= height:
            resized = (size, int(size * height / width))
        else:
            resized = (int(size * width / height), size)
        if image.size != resized:
            image = image.resize(resized, resample=Image.Resampling.BILINEAR)
        left = int(round((image.width - size) / 2.0))
        top = int(round((image.height - size) / 2.0))
        image = image.crop((left, top, left + size, top + size))
        pixels = np.asarray(image, dtype=np.float32)
    return (pixels / np.float32(255) * np.float32(255))[None]


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


def require_parity(reference, actual, name, report):
    import numpy as np

    result = metrics(reference, actual)
    result.update(atol=0.01, rtol=0.0001)
    result["passed"] = bool(np.allclose(reference, actual, atol=0.01, rtol=0.0001))
    report["parity"][name] = result
    if not result["passed"]:
        raise ValueError(f"{name} parity failed: {result}")


def signature_layout(spec, size):
    shape = spec.shape.as_list()
    if spec.dtype.name != "float32":
        raise ValueError(f"Expected float32 SavedModel signature, got {spec}")
    if shape == [1, size, size, 3]:
        return "NHWC"
    if shape == [1, 3, size, size]:
        return "NCHW"
    raise ValueError(f"SavedModel signature must be static batch 1 RGB: {spec}")


def audit_tflite(data, size):
    """Read every subgraph using public generated FlatBuffer bindings, no delegates."""
    import tflite

    model = tflite.Model.GetRootAsModel(data, 0)
    enum_names = lambda cls: {v: k for k, v in vars(cls).items() if isinstance(v, int)}
    op_names = enum_names(tflite.BuiltinOperator)
    type_names = enum_names(tflite.TensorType)
    activations = enum_names(tflite.ActivationFunctionType)
    paddings = enum_names(tflite.Padding)
    blockers, graphs, largest = [], [], None
    allowed = {"CONV_2D", "ADD", "RESIZE_NEAREST_NEIGHBOR"}
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
            option_class = {"CONV_2D": "Conv2DOptions", "ADD": "AddOptions",
                            "RESIZE_NEAREST_NEIGHBOR": "ResizeNearestNeighborOptions"}.get(name)
            if table is not None and option_class:
                obj = getattr(tflite, option_class)()
                obj.Init(table.Bytes, table.Pos)
                for field in ("Padding", "StrideH", "StrideW", "DilationHFactor", "DilationWFactor",
                              "FusedActivationFunction", "AlignCorners", "HalfPixelCenters", "PotScaleInt16"):
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
            if name == "CONV_2D" and len(inputs) == 3:
                bias_or_shape.add(inputs[2])
                data_tensors.update(inputs[:2])
                weight = graph.Tensors(inputs[1])
                kernel = [weight.Shape(k) for k in range(weight.ShapeLength())]
                if len(kernel) != 4 or kernel[1:3] != [3, 3]:
                    blockers.append(f"subgraph {g} op {j}: convolution kernel is not 3x3")
                if (options.get("Padding") != "SAME" or
                        options.get("StrideH") not in (1, 2) or
                        options.get("StrideH") != options.get("StrideW") or
                        options.get("DilationHFactor") != 1 or options.get("DilationWFactor") != 1):
                    blockers.append(f"subgraph {g} op {j}: requires zero SAME, stride 1/2, dilation 1")
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
            if tensor["shape"] != [1, size, size, 3] or tensor["dtype"] != "INT8" or len(tensor["scales"]) != 1:
                blockers.append(f"subgraph {g} IO tensor {j}: requires static NHWC [1,{size},{size},3], per-tensor int8")
        for op in ops:
            if op["name"] == "RESIZE_NEAREST_NEIGHBOR":
                if op["options"].get("AlignCorners", False):
                    blockers.append(f"subgraph {g} op {op['index']}: align_corners resize is not pixel repetition")
                src, dst = tensors[op["inputs"][0]], tensors[op["outputs"][0]]
                if src["scales"] != dst["scales"] or src["zero_points"] != dst["zero_points"]:
                    blockers.append(f"subgraph {g} op {op['index']}: resize quantization must be identical")
                s, d = src["shape"], dst["shape"]
                if len(s) != 4 or d != [s[0], s[1] * 2, s[2] * 2, s[3]]:
                    blockers.append(f"subgraph {g} op {op['index']}: resize must be nearest 2x NHWC")
        counts = {name: sum(op["name"] == name for op in ops) for name in allowed}
        if counts != {"CONV_2D": 16, "ADD": 5, "RESIZE_NEAREST_NEIGHBOR": 2}:
            blockers.append(f"subgraph {g}: expected 16 convolutions, 5 residual adds, 2 resizes; got {counts}")
        if sum(op["name"] == "CONV_2D" and op["options"].get("StrideH") == 2 for op in ops) != 2:
            blockers.append(f"subgraph {g}: expected exactly two stride-2 convolutions")
        graphs.append({"index": g, "inputs": inputs, "outputs": outputs, "operators": ops, "tensors": tensors})
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
        if len(shape) != 4 or min(shape) <= 0 or tuple(kernel) != (3, 3) or attrs.get("dilations", [1, 1]) != [1, 1]:
            raise ValueError("Expected static 3x3 convolution with dilation one")
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
    import numpy as np
    import torch
    import onnx
    import onnxruntime as ort
    import tensorflow as tf
    from neural_style import transformer_net

    if not hasattr(transformer_net, "load_checkpoint"):
        raise ValueError("Model module must expose load_checkpoint(path) and net.fused(); legacy checkpoints are unsupported")
    net, metadata = transformer_net.load_checkpoint(a.model)
    if not isinstance(metadata, dict) or not all(k in metadata for k in ("architecture", "width", "image_size")):
        raise ValueError("Checkpoint requires architecture, width and image_size metadata")
    if not isinstance(metadata["architecture"], str) or not metadata["architecture"]:
        raise ValueError("Checkpoint architecture must be a nonempty identifier")
    width = float(metadata["width"])
    if not math.isclose(width, 0.25, rel_tol=0, abs_tol=1e-9):
        raise ValueError(f"Ti60 conversion requires checkpoint width 0.25, got {width}")
    if a.width is not None and not math.isclose(a.width, width, rel_tol=0, abs_tol=1e-9):
        raise ValueError("--width differs from checkpoint metadata; it cannot change the architecture")
    a.size = a.size if a.size is not None else metadata["image_size"]
    if isinstance(a.size, bool) or not isinstance(a.size, int) or a.size < 16 or a.size % 4:
        raise ValueError("Image size must be a positive integer multiple of 4 (at least 16)")
    if a.calib_count <= 0:
        raise ValueError("--calib-count must be positive")
    files = sorted(p for p in Path(a.calib).rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:a.calib_count]
    if not files:
        raise ValueError(f"No calibration jpg/jpeg/png images under {a.calib}")
    report.update(metadata=metadata, size=a.size, calibration={"count": len(files), "files": [str(p.resolve()) for p in files],
                  "preprocessing": "RGB PIL bilinear shorter-edge Resize, CenterCrop, float32 RGB 0..255"})
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
    if (len(convs) != 16 or any(m.kernel_size != (3, 3) or m.padding_mode != "zeros" for m in convs)
            or convs[0].in_channels != 3 or convs[0].out_channels != 8 or convs[-1].out_channels != 3):
        raise ValueError("Expected width .25 RGB architecture with sixteen 3x3 zero-padded convolutions")
    image = preprocess(a.content or files[0], a.size)
    rng = np.random.default_rng(0)
    probes = [image.transpose(0, 3, 1, 2).copy(),
              rng.uniform(0, 255, (1, 3, a.size, a.size)).astype(np.float32),
              np.zeros((1, 3, a.size, a.size), dtype=np.float32)]
    references = []
    with torch.inference_mode():
        for i, x in enumerate(probes):
            ref = net(torch.from_numpy(x)).numpy()
            out = fused(torch.from_numpy(x)).numpy()
            if out.shape != x.shape:
                raise ValueError(f"Model output must preserve RGB image shape, got {out.shape}")
            require_parity(ref, out, f"pytorch_unfused_vs_fused_{i}", report)
            references.append(out)
        torch.onnx.export(fused, torch.from_numpy(probes[0]), a.onnx, opset_version=13,
                          dynamo=False, input_names=["input"], output_names=["output"],
                          dynamic_axes=None, do_constant_folding=True)
    exported = canonicalize_convolutions(onnx.load(a.onnx))
    onnx.checker.check_model(exported)
    onnx.save(exported, a.onnx)
    session = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    for i, x in enumerate(probes):
        out = session.run(None, {session.get_inputs()[0].name: x})[0]
        require_parity(references[i], out, f"pytorch_fused_vs_onnx_{i}", report)
    report["artifacts"]["onnx"] = str(Path(a.onnx).resolve())
    saved = Path(tempfile.mkdtemp(prefix="run-", dir=a.saved_model))
    report["artifacts"]["saved_model"] = str(saved.resolve())
    subprocess.run(["onnx2tf", "-i", str(Path(a.onnx).resolve()), "-o", str(saved.resolve()), "-b", "1"], check=True)
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
        require_parity(references[i], out, f"pytorch_fused_vs_saved_model_{i}", report)

    def representative_dataset():
        for path in files:
            pixels = preprocess(path, a.size)
            yield {input_name: pixels if layout == "NHWC" else pixels.transpose(0, 3, 1, 2).copy()}

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved), signature_keys=["serving_default"])
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    data = converter.convert()
    Path(a.int8_tflite).write_bytes(data)
    report["artifacts"]["int8_tflite"] = str(Path(a.int8_tflite).resolve())
    report["audit"] = audit_tflite(data, a.size)
    report["blockers"].extend(report["audit"]["blockers"])
    if report["blockers"]:
        return
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
        "target_contract": {"width": 0.25, "batch": 1, "io_layout": "NHWC", "io_dtype": "int8",
                            "convolution": "3x3 zero SAME; stride 2 on even sizes pads right/bottom by 1",
                            "residual_blocks": 5, "upsampling": "two nearest-neighbor 2x resizes"},
        "hardware_required": [
            "Nearest-neighbor 2x resize must execute in the hardware pipeline with identical input/output quantization.",
            "Pipeline boundary RGB preprocessing, scale/zero-point quantization and output dequantization/clipping need an explicit implementation and placement.",
            "Confirm compiler support for reported op versions/options, per-channel quantization, residual requantization, SAME padding and memory/scheduling on Ti60."],
        "hardware_proven": False,
        "scope": "Software parity and FlatBuffer compatibility screen only; no Ti60 compilation, placement or hardware execution proof."}
    try:
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
        Path(a.saved_model).mkdir(parents=True, exist_ok=True)
        convert(a, report)
        if not report["blockers"]:
            report["status"] = "software_checks_passed_hardware_unproven"
    except Exception as exc:
        report["blockers"].append(f"{type(exc).__name__}: {exc}")
    finally:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Report: {a.report}")
    for blocker in report["blockers"]:
        print(f"BLOCKER: {blocker}", file=sys.stderr)
    return 1 if report["blockers"] else 0


if __name__ == "__main__":
    sys.exit(main())
