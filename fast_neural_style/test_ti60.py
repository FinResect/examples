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
    BN_ARCHITECTURE, DW_ARCHITECTURE, COMPACT_ARCHITECTURE, compute_budget, expected_layers,
    diagnose_bn_folding, metrics, parity_metrics, preprocess, require_parity, validate_graph,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None


class ConversionDiagnosticsTests(unittest.TestCase):
    def run_conversion(self, conversion, extra_args=()):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "style_int8.tflite"
            arguments = ["model2tf_lite.py", "--model", str(root / "model.model"),
                         "--calib", str(root / "calib"), "--onnx", str(root / "style.onnx"),
                         "--saved-model", str(root / "saved_model"), "--int8-tflite", str(output)]
            arguments.extend(extra_args)
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

    def test_experimental_completion_is_never_strict_success(self):
        for strict_passed in (False, True):
            with self.subTest(strict_passed=strict_passed):
                def complete_experiment(args, report):
                    report["folding_experiment"] = {
                        "passed": True, "strict_folding_passed": strict_passed,
                        "probes": [{"probe_index": index} for index in range(3)],
                    }

                exit_code, report, stdout, stderr = self.run_conversion(
                    complete_experiment, ("--experimental-folding-atol", "0.1"))
                self.assertEqual(exit_code, 3)
                self.assertEqual(report["status"], "experimental_conversion_completed_hardware_unproven")
                self.assertEqual(report["stage"], "experimental_complete")
                self.assertEqual(report["mode"], "experimental_folding")
                self.assertEqual(report["experimental_folding_atol"], 0.1)
                self.assertEqual(report["blockers"], [])
                self.assertFalse(report["hardware_proven"])
                self.assertIn("not strict acceptance", stderr)

    def test_experimental_completion_requires_all_folding_probes(self):
        for count in (0, 1, 2):
            with self.subTest(count=count):
                def incomplete_experiment(args, report):
                    report["folding_experiment"] = {"passed": True, "probes": [{}] * count}

                exit_code, report, stdout, stderr = self.run_conversion(
                    incomplete_experiment, ("--experimental-folding-atol", "0.1"))
                self.assertEqual(exit_code, 1)
                self.assertEqual(report["status"], "failed")
                self.assertIn("three passing folding probes", report["blockers"][0])

    def test_experimental_option_does_not_relax_downstream_parity(self):
        for name in ("pytorch_fused_vs_onnx_0", "pytorch_fused_vs_saved_model_0"):
            with self.subTest(name=name):
                def fail_parity(args, report):
                    converter.require_export_parity(np.array([0.0]), np.array([0.05]), name, report)

                exit_code, report, stdout, stderr = self.run_conversion(
                    fail_parity, ("--experimental-folding-atol", "0.1"))
                self.assertEqual(exit_code, 1)
                self.assertEqual(report["status"], "failed")
                self.assertFalse(report["parity"][name]["passed"])
                self.assertEqual(report["parity"][name]["atol"], 0.01)

    def test_fp64_reference_completion_requires_validation_and_remains_experimental(self):
        for validated, count in ((False, 3), (True, 2), (True, 3)):
            with self.subTest(validated=validated, count=count):
                def complete_experiment(args, report):
                    report["folding_experiment"] = {"passed": True, "probes": [{}] * 3}
                    report["fp64_reference"] = {"validated": validated, "probes": [{}] * count}

                exit_code, report, stdout, stderr = self.run_conversion(
                    complete_experiment, ("--experimental-folding-atol", "0.1",
                                          "--experimental-fused-fp64-reference"))
                self.assertEqual(report["mode"], "experimental_fused_fp64_reference")
                self.assertTrue(report["experimental_fused_fp64_reference"])
                self.assertFalse(report["hardware_proven"])
                if validated and count == 3:
                    self.assertEqual(exit_code, 3)
                    self.assertEqual(report["status"], "experimental_conversion_completed_hardware_unproven")
                else:
                    self.assertEqual(exit_code, 1)
                    self.assertEqual(report["status"], "failed")

    def test_experimental_audit_blocker_still_fails(self):
        def reject_graph(args, report):
            converter.record_stage(report, "tflite_audit")
            report["blockers"].append("Unsupported operator")

        for reference_args in ((), ("--experimental-fused-fp64-reference",)):
            with self.subTest(reference_args=reference_args):
                exit_code, report, stdout, stderr = self.run_conversion(
                    reject_graph, ("--experimental-folding-atol", "0.1") + reference_args)
                self.assertEqual(exit_code, 1)
                self.assertEqual(report["status"], "failed")
                self.assertEqual(report["stage"], "tflite_audit")
                self.assertFalse(report["hardware_proven"])


class ConversionTests(unittest.TestCase):
    def test_fp64_reference_cli_requires_explicit_experimental_folding(self):
        arguments = ["model2tf_lite.py", "--model", "checkpoint.model"]
        with mock.patch.object(sys, "argv", arguments):
            self.assertFalse(converter.parse_args().experimental_fused_fp64_reference)
        with (mock.patch.object(sys, "argv", arguments + ["--experimental-fused-fp64-reference"]),
              contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error):
            converter.parse_args()
        self.assertEqual(error.exception.code, 2)
        with mock.patch.object(sys, "argv", arguments + ["--experimental-folding-atol", "0.1",
                                                       "--experimental-fused-fp64-reference"]):
            self.assertTrue(converter.parse_args().experimental_fused_fp64_reference)

    def test_experimental_folding_cli_requires_explicit_finite_budget(self):
        arguments = ["model2tf_lite.py", "--model", "checkpoint.model"]
        with mock.patch.object(sys, "argv", arguments):
            self.assertIsNone(converter.parse_args().experimental_folding_atol)
        with mock.patch.object(sys, "argv", arguments + ["--experimental-folding-atol", "0.1"]):
            self.assertEqual(converter.parse_args().experimental_folding_atol, 0.1)
        for value in ("nan", "inf", "-inf", "-1", "0", "0.009", "0.01"):
            with (self.subTest(value=value),
                  mock.patch.object(sys, "argv", arguments + [f"--experimental-folding-atol={value}"]),
                  contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error):
                converter.parse_args()
            self.assertEqual(error.exception.code, 2)

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

    def test_compact_budget_counts_weights_activations_and_vga30_requirement(self):
        budget = compute_budget(COMPACT_ARCHITECTURE, (480, 640))
        self.assertEqual(budget["macs_per_frame"], 113_356_800)
        self.assertEqual(budget["weight_bytes"], 3096)
        self.assertEqual(budget["bias_bytes"], 204)
        self.assertEqual(budget["largest_activation_bytes"], 1_228_800)
        self.assertEqual(budget["io_tensor_bytes"], {"input": 921_600, "output": 921_600})
        target = next(row for row in budget["targets"] if row["fps"] == 30)
        self.assertAlmostEqual(target["required_gmac_per_second"], 3.400704)
        self.assertEqual(compute_budget(COMPACT_ARCHITECTURE, 128)["macs_per_frame"], 6_045_696)
        self.assertEqual(compute_budget(COMPACT_ARCHITECTURE, (720, 1280))["macs_per_frame"], 340_070_400)
        self.assertEqual(compute_budget(BN_ARCHITECTURE, (480, 640))["macs_per_frame"], 2_786_918_400)
        self.assertEqual(compute_budget(BN_ARCHITECTURE, (480, 640))["largest_activation_bytes"], 4_915_200)

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

    def test_parity_reports_worst_tolerance_failure_not_largest_absolute_error(self):
        reference = np.array([0, 1000], dtype=np.float32)
        actual = np.array([0.02, 1000.05], dtype=np.float32)
        result = parity_metrics(reference, actual)
        self.assertFalse(result["passed"])
        self.assertEqual(result["failed_elements"], 1)
        self.assertEqual(result["worst_failure"]["index"], [0])
        self.assertAlmostEqual(result["worst_failure"]["abs_error"], 0.02)
        self.assertGreater(result["max_abs"], result["worst_failure"]["abs_error"])
        self.assertGreater(result["worst_failure"]["tolerance_ratio"], 1)
        json.dumps(result, allow_nan=False)

    def test_parity_diagnostics_preserve_existing_acceptance(self):
        reference = np.array([-100, 0, 1000], dtype=np.float32)
        for offset in (0, 0.009, 0.02, 1):
            with self.subTest(offset=offset):
                actual = reference + offset
                result = parity_metrics(reference, actual)
                self.assertEqual(result["passed"], bool(np.allclose(reference, actual, atol=0.01, rtol=0.0001)))
                self.assertEqual(result["failed_elements"], int(np.count_nonzero(
                    ~np.isclose(reference, actual, atol=0.01, rtol=0.0001))))
        self.assertNotIn("worst_failure", parity_metrics(reference, reference))

    def test_explicit_parity_budget_does_not_change_strict_defaults(self):
        reference, actual = np.array([0.0]), np.array([0.05])
        relaxed = parity_metrics(reference, actual, atol=0.1)
        self.assertTrue(relaxed["passed"])
        self.assertEqual(relaxed["atol"], 0.1)
        self.assertEqual(relaxed["rtol"], 0.0001)
        failed = parity_metrics(reference, np.array([0.2]), atol=0.1)
        self.assertFalse(failed["passed"])
        self.assertAlmostEqual(failed["worst_failure"]["allowed_error"], 0.10002)
        self.assertFalse(parity_metrics(reference, actual)["passed"])
        with self.assertRaisesRegex(ValueError, "parity failed"):
            require_parity(reference, actual, "unchanged", {"parity": {}})


class ExportReferenceTests(unittest.TestCase):
    def reference_report(self):
        return {"parity": {}, "fp64_reference": {
            "validated": True, "probes": [{}] * 3, "parity": {}, "strict_fp32_failures": [],
        }}

    def test_default_export_gate_remains_strict(self):
        report = {"parity": {}}
        with self.assertRaisesRegex(ValueError, "parity failed"):
            converter.require_export_parity(np.array([0.05]), np.array([0.005]), "onnx", report)
        self.assertFalse(report["parity"]["onnx"]["passed"])
        self.assertNotIn("fp64_reference", report)

    def test_reference_gate_retains_fp32_failure_with_unchanged_tolerances(self):
        for name in ("pytorch_fused_vs_onnx_0", "pytorch_fused_vs_saved_model_0"):
            with self.subTest(name=name):
                report = self.reference_report()
                converter.require_export_parity(np.array([0.05], dtype=np.float32),
                                                np.array([0.005], dtype=np.float32), name, report,
                                                np.array([0.0], dtype=np.float64))
                self.assertFalse(report["parity"][name]["passed"])
                self.assertEqual(report["fp64_reference"]["strict_fp32_failures"], [name])
                result = report["fp64_reference"]["parity"][name]
                self.assertTrue(result["passed"])
                self.assertEqual((result["atol"], result["rtol"]), (0.01, 0.0001))
                self.assertEqual(result["reference_at_fp32_worst_failure"], 0.0)
                json.dumps(report, allow_nan=False)

    def test_reference_failure_blocks_even_when_fp32_comparison_passes(self):
        report = self.reference_report()
        with self.assertRaisesRegex(ValueError, "FP64-reference parity failed"):
            converter.require_export_parity(np.array([0.0]), np.array([0.0]), "onnx", report,
                                            np.array([1.0]))
        self.assertTrue(report["parity"]["onnx"]["passed"])
        self.assertFalse(report["fp64_reference"]["parity"]["onnx"]["passed"])

    def test_reference_must_be_validated_on_all_three_probes(self):
        for validated, count in ((False, 3), (True, 1), (True, 2)):
            with self.subTest(validated=validated, count=count):
                report = self.reference_report()
                report["fp64_reference"].update(validated=validated, probes=[{}] * count)
                with self.assertRaisesRegex(ValueError, "validated three-probe"):
                    converter.require_export_parity(np.array([0.0]), np.array([0.0]), "onnx", report,
                                                    np.array([0.0]))

    def test_reference_policy_does_not_accept_nonfinite_or_mismatched_results(self):
        for actual, reference64 in ((np.array([np.nan]), np.array([0.0])),
                                    (np.array([0.0]), np.array([np.inf])),
                                    (np.array([0.0, 0.0]), np.array([0.0]))):
            with self.subTest(actual=actual, reference64=reference64), self.assertRaises(ValueError):
                converter.require_export_parity(np.array([0.0]), actual, "onnx", self.reference_report(),
                                                reference64)


class GraphValidationTests(unittest.TestCase):
    def dw_graph(self, architecture=DW_ARCHITECTURE):
        profile = expected_layers(architecture, 16, 20)
        tensors = [{"shape": [1, 16, 20, 3], "dtype": "INT8", "constant": False,
                    "scales": [0.25], "zero_points": [-3], "quantized_dimension": 0}]
        ops = []
        current = 0
        skip = None
        residual_starts = {index - 2 for index, row in enumerate(profile) if row[0] == "ADD"}
        for index, (name, h, w, ci, co, kernel, stride) in enumerate(profile):
            if index in residual_starts:
                skip = current
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
                               FusedActivationFunction="NONE" if index == len(profile) - 1 or
                               profile[index + 1][0] == "ADD" else "RELU")
                if dw:
                    options["DepthMultiplier"] = 1
            elif name == "ADD":
                inputs.append(skip)
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

    def test_compact_profile_preserves_spatial_decoder_and_residuals(self):
        tensors, ops, profile = self.dw_graph(COMPACT_ARCHITECTURE)
        self.assertEqual(profile, [
            ("CONV_2D", 8, 10, 3, 4, 3, 2),
            ("CONV_2D", 4, 5, 4, 8, 3, 2),
        ] + [
            ("CONV_2D", 4, 5, 8, 8, 3, 1),
            ("CONV_2D", 4, 5, 8, 8, 3, 1),
            ("ADD", 4, 5, 8, 8, 0, 1),
        ] * 2 + [
            ("RESIZE_NEAREST_NEIGHBOR", 8, 10, 8, 8, 0, 2),
            ("CONV_2D", 8, 10, 8, 4, 3, 1),
            ("RESIZE_NEAREST_NEIGHBOR", 16, 20, 4, 4, 0, 2),
            ("CONV_2D", 16, 20, 4, 3, 3, 1),
        ])
        blockers = []
        validate_graph(tensors, ops, profile, blockers, 0)
        self.assertEqual(blockers, [])

    def test_legacy_residual_graph_still_passes(self):
        tensors, ops, profile = self.dw_graph(BN_ARCHITECTURE)
        self.assertEqual(len(ops), 23)
        blockers = []
        validate_graph(tensors, ops, profile, blockers, 0)
        self.assertEqual(blockers, [])

    def test_compact_graph_rejects_bad_skips_and_nonlinear_residual_heads(self):
        for defect in ("skip", "second_conv", "output", "missing_block_start", "decoder"):
            with self.subTest(defect=defect):
                tensors, ops, profile = self.dw_graph(COMPACT_ARCHITECTURE)
                if defect == "skip":
                    ops[4]["inputs"][1] = ops[3]["outputs"][0]
                elif defect == "second_conv":
                    ops[3]["options"]["FusedActivationFunction"] = "RELU"
                elif defect == "output":
                    ops[-1]["options"]["FusedActivationFunction"] = "RELU"
                elif defect == "missing_block_start":
                    ops[2]["name"] = "DEPTHWISE_CONV_2D"
                else:
                    tensors[ops[-1]["inputs"][1]]["shape"] = [3, 1, 1, 4]
                blockers = []
                validate_graph(tensors, ops, profile, blockers, 0)
                self.assertTrue(blockers)

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
class FoldingExperimentTests(unittest.TestCase):
    def models_and_probes(self, fp32_shift=0.05, fp64_shift=0.0):
        """Controlled shifts exercise each gate without training or conversion dependencies."""
        import torch

        class ProbeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.folded = False
                self.register_buffer("offsets", torch.tensor([fp32_shift, fp64_shift]))

            def forward(self, inputs):
                if self.folded:
                    return inputs + self.offsets[0 if inputs.dtype == torch.float32 else 1]
                return inputs.clone()

            def fused(self):
                model = copy.deepcopy(self).eval()
                model.folded = True
                return model

        net = ProbeModel().eval()
        probes = [np.full((1, 3, 16, 20), value, dtype=np.float32) for value in (200, 0.5, 0)]
        return net, net.fused(), probes

    def test_default_success_has_no_experimental_budget(self):
        net, fused, probes = self.models_and_probes(fp32_shift=0.0)
        report = {"parity": {}}
        references = converter.check_folding_parity(net, fused, probes, report)
        self.assertEqual(len(references), 3)
        self.assertTrue(all(result["passed"] for result in report["parity"].values()))
        self.assertNotIn("folding_experiment", report)

    def test_default_still_blocks_and_collects_folding_diagnostics(self):
        net, fused, probes = self.models_and_probes()
        report = {"parity": {}}
        with (mock.patch.object(converter, "diagnose_bn_folding",
                                return_value={"comparisons": {}, "first_failed_layer": None}) as diagnose,
              self.assertRaisesRegex(ValueError, "pytorch_unfused_vs_fused_0 parity failed")):
            converter.check_folding_parity(net, fused, probes, report)
        diagnose.assert_called_once()
        self.assertFalse(report["parity"]["pytorch_unfused_vs_fused_0"]["passed"])
        self.assertEqual(report["folding_diagnostics"]["probe_index"], 0)
        self.assertNotIn("folding_experiment", report)

    def test_experiment_retains_strict_failures_and_original_fp32_outputs(self):
        net, fused, probes = self.models_and_probes()
        report = {"parity": {}}
        references = converter.check_folding_parity(net, fused, probes, report, 0.1)
        experiment = report["folding_experiment"]
        self.assertTrue(experiment["passed"])
        self.assertFalse(experiment["strict_folding_passed"])
        self.assertEqual(experiment["strict_failures"], list(report["parity"]))
        self.assertEqual(len(references), 3)
        self.assertEqual(len(experiment["probes"]), 3)
        for index, entry in enumerate(experiment["probes"]):
            strict = report["parity"][entry["strict_parity_key"]]
            self.assertFalse(strict["passed"])
            self.assertEqual(strict["atol"], 0.01)
            self.assertTrue(entry["fp32_budget"]["passed"])
            self.assertEqual(entry["fp32_budget"]["atol"], 0.1)
            self.assertTrue(entry["fp64_equivalence"]["passed"])
            self.assertEqual(entry["fp64_equivalence"]["atol"], 0.01)
            np.testing.assert_array_equal(references[index], probes[index] + np.float32(0.05))
            self.assertEqual(references[index].dtype, np.float32)
        json.dumps(report, allow_nan=False)

    def test_later_probe_budget_failures_block_after_checking_all_probes(self):
        net, fused, probes = self.models_and_probes()
        report = {"parity": {}}
        with self.assertRaisesRegex(ValueError, "experimental FP32 folding budget exceeded"):
            converter.check_folding_parity(net, fused, probes, report, 0.04)
        experiment = report["folding_experiment"]
        self.assertFalse(experiment["passed"])
        self.assertEqual(len(experiment["probes"]), 3)
        self.assertEqual([entry["fp32_budget"]["passed"] for entry in experiment["probes"]],
                         [True, False, False])
        self.assertEqual(len(experiment["failures"]), 2)

    def test_fp64_failure_blocks_even_when_experimental_fp32_budget_passes(self):
        net, fused, probes = self.models_and_probes(fp64_shift=0.05)
        report = {"parity": {}}
        with self.assertRaisesRegex(ValueError, "strict FP64 folding equivalence failed"):
            converter.check_folding_parity(net, fused, probes, report, 0.1)
        experiment = report["folding_experiment"]
        self.assertFalse(experiment["passed"])
        self.assertEqual(len(experiment["probes"]), 3)
        self.assertTrue(all(entry["fp32_budget"]["passed"] for entry in experiment["probes"]))
        self.assertTrue(all(not entry["fp64_equivalence"]["passed"] for entry in experiment["probes"]))

    def test_experiment_does_not_accept_nonfinite_outputs(self):
        net, fused, probes = self.models_and_probes(fp32_shift=float("nan"))
        report = {"parity": {}}
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            converter.check_folding_parity(net, fused, probes, report, 0.1)
        self.assertFalse(report["folding_experiment"]["passed"])


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    def test_fp64_reference_upcasts_export_weights_without_refusion_or_mutation(self):
        import torch
        from neural_style.transformer_net import DWTransformerNet

        torch.set_num_threads(1)
        torch.manual_seed(12)
        net = DWTransformerNet().eval()
        fused = net.fused()
        states = [{name: value.clone() for name, value in model.state_dict().items()}
                  for model in (net, fused)]
        probes = [np.full((1, 3, 16, 20), value, dtype=np.float32) for value in (200, 0.5, 0)]
        report = {"parity": {}}
        with mock.patch.object(fused, "fused", side_effect=AssertionError("Do not re-fuse export weights")):
            references = converter.build_fused_fp64_references(net, fused, probes, report)
        self.assertTrue(report["fp64_reference"]["validated"])
        self.assertEqual(len(report["fp64_reference"]["probes"]), 3)
        with torch.inference_mode():
            expected_model = copy.deepcopy(fused).double()
            for probe, reference in zip(probes, references):
                self.assertEqual(reference.dtype, np.float64)
                np.testing.assert_array_equal(reference, expected_model(torch.from_numpy(probe).double()).numpy())
        for model, state in zip((net, fused), states):
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state[name], atol=0, rtol=0)
            self.assertIs(next(model.parameters()).dtype, torch.float32)
            self.assertFalse(model.training)
        json.dumps(report, allow_nan=False)

    def test_fp64_reference_rejects_export_parameter_drift_on_all_probes(self):
        import torch

        net = torch.nn.Identity().eval()
        fused = torch.nn.Conv2d(3, 3, 1, groups=3).eval()
        with torch.no_grad():
            fused.weight.fill_(1)
            fused.bias.fill_(0.05)
        probes = [np.full((1, 3, 16, 20), value, dtype=np.float32) for value in (200, 0.5, 0)]
        report = {"parity": {}, "folding_experiment": {"atol": 0.1, "passed": True}}
        with self.assertRaisesRegex(ValueError, "FP32 export parameters fail strict FP64 equivalence"):
            converter.build_fused_fp64_references(net, fused, probes, report)
        details = report["fp64_reference"]
        self.assertFalse(details["validated"])
        self.assertEqual(len(details["probes"]), 3)
        self.assertEqual(len(details["failures"]), 3)
        self.assertTrue(all(entry["unfused_vs_exported_weights"]["atol"] == 0.01 for entry in details["probes"]))

    def test_experimental_folding_preserves_model_weights_dtypes_and_outputs(self):
        import torch
        from neural_style.transformer_net import DWTransformerNet

        torch.set_num_threads(1)
        torch.manual_seed(12)
        net = DWTransformerNet().eval()
        fused = net.fused()
        states = [{name: value.clone() for name, value in model.state_dict().items()}
                  for model in (net, fused)]
        probes = [np.full((1, 3, 16, 20), value, dtype=np.float32) for value in (200, 0.5, 0)]
        report = {"parity": {}}
        references = converter.check_folding_parity(net, fused, probes, report, 0.1)
        self.assertTrue(report["folding_experiment"]["passed"])
        with torch.inference_mode():
            for probe, reference in zip(probes, references):
                np.testing.assert_array_equal(reference, fused(torch.from_numpy(probe)).numpy())
        for model, state in zip((net, fused), states):
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state[name], atol=0, rtol=0)
            self.assertIs(next(model.parameters()).dtype, torch.float32)
            self.assertFalse(model.training)

    def test_folding_diagnostics_preserve_models_and_remove_hooks(self):
        import torch
        from neural_style.transformer_net import DWTransformerNet

        torch.set_num_threads(1)
        torch.manual_seed(12)
        net = DWTransformerNet().eval()
        with torch.no_grad():
            for layer in net.modules():
                if isinstance(layer, torch.nn.BatchNorm2d):
                    layer.running_mean.uniform_(-0.5, 0.5)
                    layer.running_var.uniform_(0.5, 2)
                    layer.weight.uniform_(0.5, 1.5)
                    layer.bias.uniform_(-0.2, 0.2)
            fused = net.fused()
            inputs = torch.rand(1, 3, 16, 20) * 255
            reference, actual = net(inputs).numpy(), fused(inputs).numpy()
            states = [{name: value.clone() for name, value in model.state_dict().items()}
                      for model in (net, fused)]
            diagnostics = diagnose_bn_folding(net, fused, inputs, reference, actual)
        self.assertEqual(diagnostics["input_shape"], [1, 3, 16, 20])
        self.assertEqual(len(diagnostics["layers"]), 16)
        self.assertEqual(sum("batchnorm" in layer for layer in diagnostics["layers"]), 15)
        self.assertTrue(diagnostics["comparisons"]["unfused_fp64_vs_fused_fp64"]["passed"])
        for model, state in zip((net, fused), states):
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, state[name], atol=0, rtol=0)
            self.assertFalse(any(module._forward_hooks for module in model.modules()))
        self.assertIs(next(net.parameters()).dtype, torch.float32)
        self.assertIs(next(fused.parameters()).dtype, torch.float32)
        self.assertFalse(net.training)
        self.assertFalse(fused.training)
        json.dumps(diagnostics, allow_nan=False)

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

    @unittest.skipUnless(all(importlib.util.find_spec(name) is not None for name in ("onnx", "onnxsim", "onnxruntime")),
                         "ONNX conversion dependencies are not installed")
    def test_compact_rectangular_export_preserves_residuals_and_spatial_decoder(self):
        from neural_style.transformer_net import CompactTransformerNet

        self.check_export_padding(CompactTransformerNet, (16, 20))

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
            if net.architecture == COMPACT_ARCHITECTURE:
                for name, count in (("Conv", 8), ("Add", 2), ("Resize", 2)):
                    self.assertEqual(sum(node.op_type == name for node in model.graph.node), count)
                for node in model.graph.node:
                    if node.op_type == "Conv":
                        attributes = {attribute.name: onnx.helper.get_attribute_value(attribute)
                                      for attribute in node.attribute}
                        self.assertEqual(attributes["auto_pad"], b"SAME_UPPER")
                        self.assertEqual(attributes.get("group", 1), 1)
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

    def test_compact_shapes_and_folding_with_nontrivial_bn(self):
        from neural_style.transformer_net import CompactTransformerNet

        self.check_shapes_and_folding(CompactTransformerNet, ((16, 20), (48, 64)), convolutions=8)

    def check_shapes_and_folding(self, factory, shapes, convolutions=16):
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
            self.assertEqual(sum(isinstance(m, torch.nn.Conv2d) for m in fused.modules()), convolutions)
            self.assertEqual(sum(isinstance(m, torch.nn.BatchNorm2d) for m in net.modules()), convolutions - 1)
            for height, width in shapes:
                x = torch.rand(1, 3, height, width) * 255
                ref = net(x)
                self.assertEqual(ref.shape, x.shape)
                torch.testing.assert_close(ref, fused(x), atol=0.01, rtol=0.0001)

    def test_compact_factory_and_profile_match_actual_convolutions(self):
        import torch
        from neural_style.transformer_net import CompactTransformerNet, create_model

        torch.set_num_threads(1)
        for name in ("bn_c8_r2", COMPACT_ARCHITECTURE):
            self.assertIsInstance(create_model(name), CompactTransformerNet)
            with self.assertRaisesRegex(ValueError, "width"):
                create_model(name, width=0.25)
        model = create_model("bn_c8_r2").eval()
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 3243)
        convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
        self.assertEqual(sum(module.weight.numel() for module in convolutions), 3096)
        self.assertTrue(all(module.kernel_size == (3, 3) and module.groups == 1 for module in convolutions))
        self.assertEqual(len(model.residuals), 2)
        actual = []

        def record(module, inputs, output):
            actual.append((tuple(output.shape), module.in_channels, module.stride[0],
                           output.shape[-2] * output.shape[-1] * module.weight.numel()))

        handles = [module.register_forward_hook(record) for module in convolutions]
        try:
            with torch.inference_mode():
                output = model(torch.ones(1, 3, 16, 20))
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(tuple(output.shape), (1, 3, 16, 20))
        profile = expected_layers(COMPACT_ARCHITECTURE, 16, 20)
        expected = [((1, channels_out, height, width), channels_in, stride,
                     height * width * channels_in * channels_out * kernel * kernel)
                    for name, height, width, channels_in, channels_out, kernel, stride in profile if kernel]
        self.assertEqual(actual, expected)
        self.assertEqual(sum(row[-1] for row in actual), compute_budget(COMPACT_ARCHITECTURE, (16, 20))["macs_per_frame"])

    def test_compact_decoder_is_not_forced_to_repeat_four_by_four_blocks(self):
        import torch
        from neural_style.transformer_net import CompactTransformerNet

        torch.set_num_threads(1)
        model = CompactTransformerNet().eval().fused()
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    module.weight.fill_(1 / module.weight[0].numel())
                    module.bias.fill_(0.25)
            inputs = torch.arange(3 * 16 * 20, dtype=torch.float32).reshape(1, 3, 16, 20) % 256
            output = model(inputs)
            blocks = output[:, :, 4:-4, 4:-4].reshape(1, 3, 2, 4, 3, 4)
            variation = blocks.amax(dim=(3, 5)) - blocks.amin(dim=(3, 5))
            self.assertGreater(float(variation.max()), 0.01)
            model.output.conv.weight.zero_()
            model.output.conv.bias.copy_(torch.tensor([-3., 2., 300.]))
            output = model(inputs)
            expected = torch.tensor([-3., 2., 300.]).reshape(1, 3, 1, 1).expand_as(output)
            torch.testing.assert_close(output, expected)

    def test_compact_training_backward_reaches_all_parameters(self):
        import torch
        from neural_style.transformer_net import CompactTransformerNet

        torch.set_num_threads(1)
        torch.manual_seed(12)
        model = CompactTransformerNet().train()
        output = model(torch.rand(2, 3, 16, 20) * 255)
        output.square().mean().backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_training_preview_preserves_batchnorm_state_and_training_modes(self):
        import torch
        from neural_style import utils
        from neural_style.transformer_net import CompactTransformerNet

        torch.set_num_threads(1)
        model = CompactTransformerNet().train()
        model.conv1.bn.eval()
        modes = [module.training for module in model.modules()]
        state = copy.deepcopy(model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "preview.png"
            utils.save_eval_preview(model, torch.ones(1, 3, 16, 20), path)
            with Image.open(path) as preview:
                self.assertEqual(preview.size, (20, 16))
                self.assertEqual(preview.mode, "RGB")
        self.assertEqual([module.training for module in model.modules()], modes)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)

    def test_training_preview_failure_restores_modes_and_rejects_nonfinite(self):
        import torch
        from neural_style import utils
        from neural_style.transformer_net import CompactTransformerNet

        torch.set_num_threads(1)
        model = CompactTransformerNet().train()
        model.conv1.bn.eval()
        modes = [module.training for module in model.modules()]
        inputs = torch.ones(1, 3, 16, 20)
        with mock.patch.object(utils, "save_image", side_effect=OSError("cannot save")):
            with self.assertRaisesRegex(OSError, "cannot save"):
                utils.save_eval_preview(model, inputs, "unused.png")
        self.assertEqual([module.training for module in model.modules()], modes)
        with torch.no_grad():
            model.output.conv.bias.fill_(float("nan"))
        with mock.patch.object(utils, "save_image") as save:
            with self.assertRaisesRegex(RuntimeError, "Non-finite"):
                utils.save_eval_preview(model, inputs, "unused.png")
            save.assert_not_called()
        self.assertEqual([module.training for module in model.modules()], modes)

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

    def test_compact_checkpoint_roundtrip_and_architecture_isolation(self):
        import torch
        from neural_style.transformer_net import (
            CompactTransformerNet, checkpoint_dimensions, load_checkpoint, save_checkpoint,
        )

        torch.set_num_threads(1)
        model = CompactTransformerNet().eval()
        model.conv1.bn.running_mean.fill_(7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compact.model"
            for size in (128, (16, 20)):
                with self.subTest(size=size):
                    save_checkpoint(path, model, size, epoch=2)
                    restored, metadata = load_checkpoint(path)
                    self.assertIsInstance(restored, CompactTransformerNet)
                    self.assertEqual(metadata["architecture"], COMPACT_ARCHITECTURE)
                    self.assertNotIn("width", metadata)
                    self.assertEqual(checkpoint_dimensions(metadata), (128, 128) if size == 128 else size)
                    self.assertFalse(any(module.training for module in restored.modules()))
                    for name, value in model.state_dict().items():
                        torch.testing.assert_close(restored.state_dict()[name], value)
                    with torch.inference_mode():
                        inputs = torch.ones(1, 3, 16, 20)
                        torch.testing.assert_close(model(inputs), restored(inputs))
            checkpoint = torch.load(path, weights_only=True)
            checkpoint["metadata"].update(architecture=BN_ARCHITECTURE, width=0.25)
            torch.save(checkpoint, path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(path)
        with self.assertRaisesRegex(ValueError, "requires image_height"):
            checkpoint_dimensions({"architecture": COMPACT_ARCHITECTURE, "image_size": 128})
        with self.assertRaisesRegex(ValueError, "width"):
            checkpoint_dimensions({"architecture": COMPACT_ARCHITECTURE, "image_size": 128,
                                   "image_height": 128, "image_width": 128, "width": 0.25})


if __name__ == "__main__":
    unittest.main()
