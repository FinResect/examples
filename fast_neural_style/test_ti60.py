"""No training or downloads: python -m unittest discover -s fast_neural_style -p 'test_ti60.py'."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from script import model2tf_lite as converter
from script.model2tf_lite import (
    BN_ARCHITECTURE, DW_ARCHITECTURE, compute_budget, expected_layers,
    metrics, preprocess, require_parity, validate_graph,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ConversionDiagnosticsTests(unittest.TestCase):
    def run_conversion(self, conversion):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "style_int8.tflite"
            arguments = ["model2tf_lite.py", "--model", str(root / "model.model"),
                         "--calib", str(root / "calib"), "--onnx", str(root / "style.onnx"),
                         "--saved-model", str(root / "saved_model"), "--int8-tflite", str(output)]
            stdout, stderr = io.StringIO(), io.StringIO()
            with (mock.patch.object(sys, "argv", arguments),
                  mock.patch.object(converter, "convert", side_effect=conversion),
                  contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr)):
                exit_code = converter.main()
            report = json.loads(Path(str(output) + ".report.json").read_text(encoding="utf-8"))
        return exit_code, report, stdout.getvalue(), stderr.getvalue()

    def test_recursion_failure_preserves_stage_and_traceback(self):
        def fail_conversion(args, report):
            converter.record_stage(report, "import_tensorflow")
            raise RecursionError("simulated import recursion")

        exit_code, report, stdout, stderr = self.run_conversion(fail_conversion)
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stage"], "import_tensorflow")
        self.assertEqual(report["blockers"], ["RecursionError: simulated import recursion"])
        self.assertEqual(report["parity"], {})
        self.assertEqual(report["artifacts"], {})
        self.assertFalse(report["hardware_proven"])
        self.assertIn("Traceback (most recent call last):", report["traceback"])
        self.assertIn("fail_conversion", report["traceback"])
        self.assertIn(report["blockers"][0], report["traceback"])
        self.assertIn("[stage] import_tensorflow", stderr)
        self.assertIn(report["traceback"], stderr)
        self.assertIn("Report:", stdout)

    def test_success_marks_completion_without_hardware_proof(self):
        exit_code, report, stdout, stderr = self.run_conversion(lambda args, report: None)
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["status"], "software_checks_passed_hardware_unproven")
        self.assertEqual(report["stage"], "complete")
        self.assertEqual(report["blockers"], [])
        self.assertFalse(report["hardware_proven"])
        self.assertNotIn("traceback", report)
        self.assertIn("[stage] complete", stderr)
        self.assertIn("Report:", stdout)

    def test_audit_blocker_preserves_failed_stage_without_traceback(self):
        def reject_graph(args, report):
            converter.record_stage(report, "tflite_audit")
            report["blockers"].append("Unsupported operator")

        exit_code, report, stdout, stderr = self.run_conversion(reject_graph)
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["stage"], "tflite_audit")
        self.assertFalse(report["hardware_proven"])
        self.assertNotIn("traceback", report)
        self.assertIn("BLOCKER: Unsupported operator", stderr)
        self.assertNotIn("[stage] complete", stderr)
        self.assertIn("Report:", stdout)


class ConversionTests(unittest.TestCase):
    def test_preprocess_center_crop_preserves_rgb_range(self):
        image = np.zeros((16, 32, 3), dtype=np.uint8)
        image[:, 8:24] = [12, 128, 255]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "content.png"
            Image.fromarray(image).save(path)
            result = preprocess(path, 16)
        self.assertEqual(result.shape, (1, 16, 16, 3))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, np.broadcast_to([12, 128, 255], result.shape), atol=1e-5)

    def test_shared_preprocess_square_and_rectangular_rgb(self):
        from neural_style.preprocessing import resize_crop_image

        # Distinct spatial/channel values expose swapped axes, BGR and normalization.
        y, x = np.indices((24, 40))
        pixels = np.stack((x * 6, y * 10, (x + y) * 4), axis=-1).astype(np.uint8)
        source = Image.fromarray(pixels)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.png"
            source.save(path)
            for size, shape, box in ((16, (16, 16), (5, 0, 21, 16)),
                                     ((16, 20), (16, 20), (3, 0, 23, 16)),
                                     ((32, 16), (32, 16), (18, 0, 34, 32))):
                with self.subTest(size=size):
                    # Independently specified aspect-fill resize and center crop.
                    resized = (26, 16) if shape[0] == 16 else (53, 32)
                    expected = np.asarray(source.resize(resized, Image.Resampling.BILINEAR).crop(box))
                    shared = resize_crop_image(source, size)
                    result = preprocess(path, size)
                    self.assertEqual(shared.mode, "RGB")
                    self.assertEqual(result.shape, (1, *shape, 3))
                    self.assertEqual(result.dtype, np.float32)
                    np.testing.assert_array_equal(np.asarray(shared), expected)
                    np.testing.assert_array_equal(result[0], expected.astype(np.float32))
            np.testing.assert_array_equal(preprocess(path, 16), preprocess(path, (16, 16)))

    def test_preprocess_converts_grayscale_and_rgba_to_rgb(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode, color, rgb in (("L", 91, [91, 91, 91]),
                                     ("RGBA", (12, 128, 255, 0), [12, 128, 255])):
                with self.subTest(mode=mode):
                    path = Path(directory) / (mode + ".png")
                    Image.new(mode, (40, 24), color).save(path)
                    result = preprocess(path, (16, 20))
                    self.assertEqual(result.shape, (1, 16, 20, 3))
                    np.testing.assert_array_equal(result, np.broadcast_to(rgb, result.shape))

    def test_dw_budget_vga_720p_and_rectangular_scaling(self):
        budget = compute_budget(DW_ARCHITECTURE, (480, 640))
        self.assertEqual(budget["macs_per_frame"], 12_979_200)
        targets = {(row["height"], row["width"], row["fps"]): row for row in budget["targets"]}
        self.assertAlmostEqual(targets[480, 640, 60]["required_gmac_per_second"], 0.778752)
        self.assertEqual(targets[720, 1280, 60]["macs_per_frame"], 38_937_600)
        self.assertAlmostEqual(targets[720, 1280, 60]["required_gmac_per_second"], 2.336256)
        self.assertEqual(compute_budget(DW_ARCHITECTURE, (720, 1280))["macs_per_frame"], 38_937_600)
        self.assertEqual(compute_budget(DW_ARCHITECTURE, (48, 64))["macs_per_frame"], 129_792)
        self.assertEqual(compute_budget(BN_ARCHITECTURE, 128)["macs_per_frame"], 148_635_648)
        for layer in budget["layers"]:
            if layer["name"] == "RESIZE_NEAREST_NEIGHBOR":
                self.assertEqual((layer["macs"], layer["weight_bytes"]), (0, 0))

    def test_budget_rejects_invalid_dimensions_and_architecture(self):
        for size in (15, 18, (16, 19), (True, 20), (16.0, 20)):
            with self.subTest(size=size), self.assertRaises(ValueError):
                compute_budget(DW_ARCHITECTURE, size)
        with self.assertRaisesRegex(ValueError, "architecture"):
            compute_budget("unknown", (16, 20))

    def test_parity_rejects_bad_results(self):
        report = {"parity": {}}
        reference = np.zeros((1, 3, 4, 4), dtype=np.float32)
        with self.assertRaises(ValueError):
            require_parity(reference, reference + 1, "changed", report)
        self.assertFalse(report["parity"]["changed"]["passed"])
        with self.assertRaises(ValueError):
            metrics(reference, np.full_like(reference, np.nan))
        with self.assertRaises(ValueError):
            metrics(reference, reference[:, :, :2])


class GraphValidationTests(unittest.TestCase):
    def dw_graph(self):
        profile = expected_layers(DW_ARCHITECTURE, 16, 20)
        tensors = [{"shape": [1, 16, 20, 3], "dtype": "INT8", "constant": False,
                    "scales": [0.25], "zero_points": [-3], "quantized_dimension": 0}]
        ops = []
        current = 0
        for index, (name, h, w, ci, co, kernel, stride) in enumerate(profile):
            inputs = [current]
            options = {"FusedActivationFunction": "NONE"}
            if kernel:
                dw = name == "DEPTHWISE_CONV_2D"
                scales = [0.125 + channel / 32 for channel in range(co)]
                inputs.extend([len(tensors), len(tensors) + 1])
                tensors.extend([
                    {"shape": [1, kernel, kernel, co] if dw else [co, kernel, kernel, ci],
                     "dtype": "INT8", "constant": True, "scales": scales,
                     "zero_points": [0] * co, "quantized_dimension": 3 if dw else 0},
                    {"shape": [co], "dtype": "INT32", "constant": True,
                     "scales": [tensors[current]["scales"][0] * scale for scale in scales],
                     "zero_points": [0] * co, "quantized_dimension": 0},
                ])
                options.update(StrideH=stride, StrideW=stride, Padding="SAME",
                               DilationHFactor=1, DilationWFactor=1,
                               FusedActivationFunction="NONE" if index == len(profile) - 1 else "RELU")
                if dw:
                    options["DepthMultiplier"] = 1
            else:
                inputs.append(len(tensors))
                tensors.append({"shape": [2], "dtype": "INT32", "constant": True,
                                "scales": [], "zero_points": [], "quantized_dimension": 0})
                options.update(AlignCorners=False, HalfPixelCenters=False)
            output = len(tensors)
            tensors.append({"shape": [1, h, w, co], "dtype": "INT8", "constant": False,
                            "scales": [0.25], "zero_points": [-3], "quantized_dimension": 0})
            ops.append({"name": name, "inputs": inputs, "outputs": [output], "options": options})
            current = output
        return tensors, ops, profile

    def test_dw_profile_order_counts_and_valid_quantized_pipeline(self):
        tensors, ops, profile = self.dw_graph()
        expected = [
            ("DEPTHWISE_CONV_2D", 8, 10, 3, 3, 3, 2), ("CONV_2D", 8, 10, 3, 4, 1, 1),
            ("DEPTHWISE_CONV_2D", 4, 5, 4, 4, 3, 2), ("CONV_2D", 4, 5, 4, 4, 1, 1),
        ] + [("DEPTHWISE_CONV_2D", 4, 5, 4, 4, 3, 1), ("CONV_2D", 4, 5, 4, 4, 1, 1)] * 5 + [
            ("CONV_2D", 4, 5, 4, 4, 1, 1),
            ("RESIZE_NEAREST_NEIGHBOR", 8, 10, 4, 4, 0, 2),
            ("RESIZE_NEAREST_NEIGHBOR", 16, 20, 4, 4, 0, 2),
            ("CONV_2D", 16, 20, 4, 3, 1, 1),
        ]
        self.assertEqual(profile, expected)
        self.assertEqual([sum(row[0] == name for row in profile) for name in
                          ("DEPTHWISE_CONV_2D", "CONV_2D", "RESIZE_NEAREST_NEIGHBOR", "ADD")],
                         [7, 9, 2, 0])
        blockers = []
        validate_graph(tensors, ops, profile, blockers, 0)
        self.assertEqual(blockers, [])

    def test_dw_quantization_and_constant_contract_rejections(self):
        cases = (
            (0, 1, "quantized_dimension", 0, "per-output-channel INT8 weights"),
            (1, 1, "quantized_dimension", 3, "per-output-channel INT8 weights"),
            (0, 1, "constant", False, "per-output-channel INT8 weights"),
            (0, 1, "zero_points", [1, 0, 0], "per-output-channel INT8 weights"),
            (0, 1, "scales", [0.125], "per-output-channel INT8 weights"),
            (0, 2, "dtype", "INT8", "per-channel INT32 bias"),
            (0, 2, "constant", False, "per-channel INT32 bias"),
            (0, 2, "quantized_dimension", 3, "per-channel INT32 bias"),
            (0, 2, "scales", [1.0, 1.0, 1.0], "bias scales differ"),
            (0, 0, "scales", [0.25, 0.5], "per-tensor INT8"),
        )
        for index, slot, field, value, message in cases:
            with self.subTest(op=index, slot=slot, field=field):
                tensors, ops, profile = self.dw_graph()
                tensors[ops[index]["inputs"][slot]][field] = value
                blockers = []
                validate_graph(tensors, ops, profile, blockers, 0)
                self.assertTrue(any(message in blocker for blocker in blockers), blockers)

    def test_dw_pipeline_rejects_rewiring_order_count_and_nonlinear_head(self):
        for defect, message in (("rewire", "declared pipeline"), ("order", "expected DEPTHWISE_CONV_2D"),
                                ("count", "expected 18 operators"), ("head", "must be linear"),
                                ("hidden", "unexpected activation"), ("shape", "feature shapes")):
            with self.subTest(defect=defect):
                tensors, ops, profile = self.dw_graph()
                if defect == "rewire":
                    ops[4]["inputs"][0] = ops[2]["inputs"][0]
                elif defect == "order":
                    ops[0], ops[1] = ops[1], ops[0]
                elif defect == "count":
                    ops.pop()
                elif defect == "head":
                    ops[-1]["options"]["FusedActivationFunction"] = "RELU"
                elif defect == "hidden":
                    ops[0]["options"]["FusedActivationFunction"] = "NONE"
                else:
                    tensors[ops[-1]["outputs"][0]]["shape"] = [1, 20, 16, 3]
                blockers = []
                validate_graph(tensors, ops, profile, blockers, 0)
                self.assertTrue(any(message in blocker for blocker in blockers), blockers)


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(m) is not None for m in ("onnx", "onnxsim", "onnxruntime")),
                         "ONNX conversion dependencies are not installed")
    def test_export_canonicalizes_padding_without_changing_values(self):
        from neural_style.transformer_net import TransformerNet

        self.check_export_padding(TransformerNet, (16, 16))

    @unittest.skipUnless(all(importlib.util.find_spec(m) is not None for m in ("onnx", "onnxsim", "onnxruntime")),
                         "ONNX conversion dependencies are not installed")
    def test_dw_rectangular_export_canonicalizes_depthwise_and_pointwise_padding(self):
        from neural_style.transformer_net import DWTransformerNet

        self.check_export_padding(DWTransformerNet, (16, 20), depthwise=True)

    def check_export_padding(self, factory, shape, depthwise=False):
        import torch
        import onnx
        import onnxruntime
        from script.model2tf_lite import canonicalize_convolutions

        torch.set_num_threads(1)
        torch.manual_seed(12)
        net = factory().eval().fused()
        if depthwise:
            # Keep spatial signal alive through all 16 convolutions so padding errors matter.
            with torch.no_grad():
                for layer in net.modules():
                    if isinstance(layer, torch.nn.Conv2d):
                        layer.weight.fill_(1 / layer.weight[0].numel())
                        layer.bias.fill_(0.25)
        x = torch.arange(3 * shape[0] * shape[1], dtype=torch.float32).reshape(1, 3, *shape) % 256
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            path = Path(directory) / "test.onnx"
            torch.onnx.export(net, x, path, opset_version=13, dynamo=False,
                              input_names=["input"], output_names=["output"])
            model = canonicalize_convolutions(onnx.load(path))
            self.assertFalse(any(node.op_type == "Pad" for node in model.graph.node))
            if depthwise:
                weights = {item.name: onnx.numpy_helper.to_array(item) for item in model.graph.initializer}
                convs = [node for node in model.graph.node if node.op_type == "Conv"]
                self.assertEqual(len(convs), 16)
                groups = []
                kernels = []
                for node in convs:
                    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
                    self.assertEqual(attrs["auto_pad"], b"SAME_UPPER")
                    self.assertNotIn("pads", attrs)
                    groups.append(attrs.get("group", 1))
                    kernels.append(tuple(weights[node.input[1]].shape[2:]))
                self.assertEqual(groups, [3, 1] + [4, 1] * 6 + [1, 1])
                self.assertEqual(kernels, [(3, 3), (1, 1)] * 7 + [(1, 1)] * 2)
                self.assertFalse(any(node.op_type == "Add" for node in model.graph.node))
            session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            output = session.run(None, {"input": x.numpy()})[0]
            self.assertEqual(output.shape, tuple(x.shape))
            reference = net(x).numpy()
            if depthwise:
                self.assertGreater(float(np.ptp(reference)), 1.0)
            np.testing.assert_allclose(reference, output, atol=0.01, rtol=0.0001)

    def test_shapes_and_folding_with_nontrivial_bn(self):
        from neural_style.transformer_net import TransformerNet

        self.check_shapes_and_folding(TransformerNet, ((16, 16), (128, 128)))

    def test_dw_rectangular_shapes_and_folding_with_nontrivial_bn(self):
        from neural_style.transformer_net import DWTransformerNet

        self.check_shapes_and_folding(DWTransformerNet, ((16, 20), (48, 64)))

    def check_shapes_and_folding(self, factory, shapes):
        import torch

        torch.set_num_threads(1)
        torch.manual_seed(12)
        net = factory().eval()
        with torch.no_grad():
            for layer in net.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.running_mean.uniform_(-0.5, 0.5)
                    layer.running_var.uniform_(0.5, 2)
                    layer.weight.uniform_(0.5, 1.5)
                    layer.bias.uniform_(-0.2, 0.2)
            fused = net.fused()
            self.assertIsNot(fused, net)
            self.assertFalse(any(module.training for module in fused.modules()))
            self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules()))
            self.assertEqual(sum(isinstance(m, torch.nn.Conv2d) for m in fused.modules()), 16)
            self.assertEqual(sum(isinstance(m, torch.nn.BatchNorm2d) for m in net.modules()), 15)
            for height, width in shapes:
                x = torch.rand(1, 3, height, width) * 255
                ref = net(x)
                self.assertEqual(ref.shape, x.shape)
                torch.testing.assert_close(ref, fused(x), atol=0.01, rtol=0.0001)

    def test_default_dw_factory_structure_and_linear_rgb_head(self):
        import operator
        import torch
        from torch.fx import symbolic_trace
        from neural_style.transformer_net import ConvLayer, DWTransformerNet, TransformerNet, create_model

        torch.set_num_threads(1)
        net = create_model().eval()
        self.assertIsInstance(net, DWTransformerNet)
        for name in ("dw_c4_s", DW_ARCHITECTURE):
            self.assertIsInstance(create_model(name), DWTransformerNet)
        for name in ("bn_3x3", BN_ARCHITECTURE):
            self.assertIsInstance(create_model(name), TransformerNet)
            self.assertEqual(create_model(name, width=0.5).conv1.conv.out_channels, 16)
        with self.assertRaisesRegex(ValueError, "width"):
            create_model(width=0.25)
        with self.assertRaisesRegex(ValueError, "architecture"):
            create_model("unknown")
        layers = list(net.layers)
        self.assertEqual([type(layer) for layer in layers],
                         [ConvLayer, torch.nn.ReLU, ConvLayer, torch.nn.ReLU] * 7 +
                         [ConvLayer, torch.nn.ReLU, torch.nn.Upsample, torch.nn.Upsample, ConvLayer])
        convs = [layer.conv for layer in layers if isinstance(layer, ConvLayer)]
        self.assertEqual([(m.in_channels, m.out_channels, m.groups, m.kernel_size, m.stride) for m in convs],
                         [(3, 3, 3, (3, 3), (2, 2)), (3, 4, 1, (1, 1), (1, 1)),
                          (4, 4, 4, (3, 3), (2, 2)), (4, 4, 1, (1, 1), (1, 1))] +
                         [(4, 4, 4, (3, 3), (1, 1)), (4, 4, 1, (1, 1), (1, 1))] * 5 +
                         [(4, 4, 1, (1, 1), (1, 1)), (4, 3, 1, (1, 1), (1, 1))])
        self.assertTrue(all(layer.mode == "nearest" and layer.scale_factor == 2
                            for layer in layers if isinstance(layer, torch.nn.Upsample)))
        self.assertIsInstance(layers[-1].bn, torch.nn.Identity)
        graph = symbolic_trace(net).graph
        self.assertFalse(any(node.op == "call_function" and node.target in (operator.add, torch.add)
                             for node in graph.nodes))
        with torch.no_grad():
            convs[-1].weight.zero_()
            convs[-1].bias.copy_(torch.tensor([-3., -1., 2.]))
            output = net(torch.zeros(1, 3, 16, 20))
        torch.testing.assert_close(output, torch.tensor([-3., -1., 2.]).reshape(1, 3, 1, 1).expand_as(output))

    def test_stride_two_same_boundary(self):
        import torch
        from neural_style.transformer_net import ConvLayer

        layer = ConvLayer(1, 1, stride=2, normalize=False)
        with torch.no_grad():
            layer.conv.weight.fill_(1)
            layer.conv.bias.zero_()
            result = layer(torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4))
        torch.testing.assert_close(result, torch.tensor([[[[45., 39.], [66., 50.]]]]))

    def test_checkpoint_retains_bn_and_rejects_legacy(self):
        import torch
        from neural_style.transformer_net import TransformerNet, load_checkpoint, save_checkpoint

        model = TransformerNet()
        model.conv1.bn.running_mean.fill_(7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.model"
            save_checkpoint(path, model, 128)
            restored, metadata = load_checkpoint(path)
            self.assertEqual(metadata["width"], 0.25)
            self.assertFalse(restored.training)
            torch.testing.assert_close(restored.conv1.bn.running_mean, model.conv1.bn.running_mean)
            torch.save(model.state_dict(), path)
            with self.assertRaisesRegex(ValueError, "Legacy"):
                load_checkpoint(path)

    def test_dw_rectangular_checkpoint_roundtrip_and_strict_state(self):
        import torch
        from neural_style.transformer_net import (
            DWTransformerNet, TransformerNet, checkpoint_dimensions, load_checkpoint, save_checkpoint,
        )

        torch.set_num_threads(1)
        torch.manual_seed(12)
        net = DWTransformerNet().eval()
        net.layers[0].bn.running_mean.copy_(torch.tensor([-0.5, 0.25, 1.5]))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dw.model"
            save_checkpoint(path, net, (16, 20), epoch=7)
            restored, metadata = load_checkpoint(path)
            self.assertIsInstance(restored, DWTransformerNet)
            self.assertEqual(metadata, {"architecture": DW_ARCHITECTURE, "image_size": (16, 20),
                                        "image_height": 16, "image_width": 20})
            self.assertEqual(checkpoint_dimensions(metadata), (16, 20))
            self.assertFalse(any(module.training for module in restored.modules()))
            self.assertTrue(all(value.device.type == "cpu" for value in restored.state_dict().values()))
            self.assertEqual(restored.state_dict().keys(), net.state_dict().keys())
            for key, value in net.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)
            with torch.no_grad():
                x = torch.rand(1, 3, 16, 20) * 255
                torch.testing.assert_close(restored(x), net(x))
            checkpoint = torch.load(path, weights_only=True)
            self.assertEqual(checkpoint["epoch"], 7)
            for defect in ("missing", "unexpected", "shape", "architecture"):
                with self.subTest(defect=defect):
                    broken = copy.deepcopy(checkpoint)
                    state = broken["state_dict"]
                    if defect == "missing":
                        del state["layers.0.conv.weight"]
                    elif defect == "unexpected":
                        state["unexpected.weight"] = torch.zeros(1)
                    elif defect == "shape":
                        state["layers.0.conv.weight"] = torch.zeros(4, 1, 3, 3)
                    else:
                        broken["metadata"]["architecture"] = BN_ARCHITECTURE
                        broken["metadata"]["width"] = 0.25
                    torch.save(broken, path)
                    with self.assertRaises(RuntimeError):
                        load_checkpoint(path)
            for legacy in (net.state_dict(), TransformerNet().state_dict()):
                torch.save(legacy, path)
                with self.assertRaisesRegex(ValueError, "Legacy"):
                    load_checkpoint(path)

    def test_checkpoint_dimensions_reject_inconsistent_rectangular_metadata(self):
        from neural_style.transformer_net import checkpoint_dimensions

        self.assertEqual(checkpoint_dimensions({"architecture": BN_ARCHITECTURE, "image_size": 128}),
                         (128, 128))
        valid = {"architecture": DW_ARCHITECTURE, "image_size": (16, 20),
                 "image_height": 16, "image_width": 20}
        for changes in ({"image_size": 16}, {"image_width": 24},
                        {"image_height": 18, "image_size": (18, 20)},
                        {"architecture": "unknown"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                checkpoint_dimensions({**valid, **changes})
        with self.assertRaisesRegex(ValueError, "requires image_height"):
            checkpoint_dimensions({"architecture": DW_ARCHITECTURE, "image_size": 16})


if __name__ == "__main__":
    unittest.main()
