"""No training or downloads: python -m unittest discover -s fast_neural_style -p 'test_ti60.py'."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from script.model2tf_lite import metrics, preprocess, require_parity

HAS_TORCH = importlib.util.find_spec("torch") is not None


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


@unittest.skipUnless(HAS_TORCH, "PyTorch is not installed")
class ModelTests(unittest.TestCase):
    @unittest.skipUnless(all(importlib.util.find_spec(m) is not None for m in ("onnx", "onnxsim", "onnxruntime")),
                         "ONNX conversion dependencies are not installed")
    def test_export_canonicalizes_padding_without_changing_values(self):
        import torch
        import onnx
        import onnxruntime
        from neural_style.transformer_net import TransformerNet
        from script.model2tf_lite import canonicalize_convolutions

        torch.set_num_threads(1)
        net = TransformerNet().eval().fused()
        x = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16) % 256
        with tempfile.TemporaryDirectory() as directory, torch.no_grad():
            path = Path(directory) / "test.onnx"
            torch.onnx.export(net, x, path, opset_version=13, dynamo=False,
                              input_names=["input"], output_names=["output"])
            model = canonicalize_convolutions(onnx.load(path))
            self.assertFalse(any(node.op_type == "Pad" for node in model.graph.node))
            session = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
            output = session.run(None, {"input": x.numpy()})[0]
            np.testing.assert_allclose(net(x).numpy(), output, atol=0.01, rtol=0.0001)

    def test_shapes_and_folding_with_nontrivial_bn(self):
        import torch
        from neural_style.transformer_net import TransformerNet

        torch.set_num_threads(1)
        torch.manual_seed(12)
        net = TransformerNet().eval()
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
            for size in (16, 128):
                x = torch.rand(1, 3, size, size) * 255
                ref = net(x)
                self.assertEqual(ref.shape, x.shape)
                torch.testing.assert_close(ref, fused(x), atol=0.01, rtol=0.0001)

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


if __name__ == "__main__":
    unittest.main()
