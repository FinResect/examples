import copy
import math

import torch
from torch import nn

try:
    from stream_net import (
        STREAM_ARCHITECTURE,
        StreamConv,
        StreamStudentNet,
        load_stream_checkpoint,
        save_stream_checkpoint,
    )
except (ImportError, ModuleNotFoundError):
    from neural_style.stream_net import (
        STREAM_ARCHITECTURE,
        StreamConv,
        StreamStudentNet,
        load_stream_checkpoint,
        save_stream_checkpoint,
    )


ARCHITECTURE = "ti60_style_bn_3x3_v1"


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, normalize=True):
        super().__init__()
        # For even inputs, TF SAME stride=2 pads only the right and bottom.
        self.pad = nn.ZeroPad2d((0, 1, 0, 1)) if stride == 2 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, stride,
                              padding=1 if stride == 1 else 0)
        self.bn = nn.BatchNorm2d(out_channels) if normalize else nn.Identity()

    def forward(self, x):
        return self.bn(self.conv(self.pad(x)))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvLayer(channels, channels)
        self.conv2 = ConvLayer(channels, channels)

    def forward(self, x):
        return x + self.conv2(torch.relu(self.conv1(x)))


class TransformerNet(nn.Module):
    def __init__(self, width=0.25):
        super().__init__()
        if not math.isfinite(width) or width <= 0 or int(32 * width) < 1:
            raise ValueError("width must produce at least one channel")
        self.width = width
        c1, c2, c3 = (int(c * width) for c in (32, 64, 128))
        self.conv1 = ConvLayer(3, c1)
        self.conv2 = ConvLayer(c1, c2, stride=2)
        self.conv3 = ConvLayer(c2, c3, stride=2)
        self.residuals = nn.Sequential(*(ResidualBlock(c3) for _ in range(5)))
        self.up1 = ConvLayer(c3, c2)
        self.up2 = ConvLayer(c2, c1)
        self.output = ConvLayer(c1, 3, normalize=False)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.residuals(torch.relu(self.conv3(x)))
        x = torch.relu(self.up1(nn.functional.interpolate(x, scale_factor=2, mode="nearest")))
        x = torch.relu(self.up2(nn.functional.interpolate(x, scale_factor=2, mode="nearest")))
        return self.output(x)

    def fused(self):
        model = copy.deepcopy(self).eval()
        for layer in model.modules():
            if isinstance(layer, ConvLayer) and isinstance(layer.bn, nn.BatchNorm2d):
                layer.conv = nn.utils.fusion.fuse_conv_bn_eval(layer.conv, layer.bn)
                layer.bn = nn.Identity()
        return model.eval()


def save_checkpoint(path, model, image_size, **extra):
    torch.save({
        "metadata": {"architecture": ARCHITECTURE, "width": model.width,
                     "image_size": image_size},
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        **extra,
    }, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "metadata" not in checkpoint:
        raise ValueError("Legacy InstanceNorm/raw state_dict models are incompatible; retrain the Ti60 model")
    metadata = checkpoint["metadata"]
    if metadata.get("architecture") == STREAM_ARCHITECTURE:
        return load_stream_checkpoint(path)
    if metadata.get("architecture") != ARCHITECTURE:
        raise ValueError(f"Unsupported checkpoint architecture: {metadata.get('architecture')}")
    size = metadata.get("image_size")
    if not isinstance(size, int) or size < 16 or size % 4:
        raise ValueError("Checkpoint image_size must be >=16 and divisible by four")
    model = TransformerNet(metadata["width"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), metadata


class LegacyConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.reflection_pad = nn.ReflectionPad2d(kernel_size // 2)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv2d(self.reflection_pad(x))


class LegacyResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = LegacyConvLayer(channels, channels, 3, 1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = LegacyConvLayer(channels, channels, 3, 1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        return x + self.in2(self.conv2(self.relu(self.in1(self.conv1(x)))))


class LegacyTransformerNet(nn.Module):
    """Legacy InstanceNorm TransformerNet for loading pre-existing teacher checkpoints."""
    def __init__(self, width=1.0):
        super().__init__()
        self.width = width
        c1, c2, c3 = int(32 * width), int(64 * width), int(128 * width)
        self.conv1 = LegacyConvLayer(3, c1, 9, 1)
        self.in1 = nn.InstanceNorm2d(c1, affine=True)
        self.conv2 = LegacyConvLayer(c1, c2, 3, 2)
        self.in2 = nn.InstanceNorm2d(c2, affine=True)
        self.conv3 = LegacyConvLayer(c2, c3, 3, 2)
        self.in3 = nn.InstanceNorm2d(c3, affine=True)
        self.res1 = LegacyResidualBlock(c3)
        self.res2 = LegacyResidualBlock(c3)
        self.res3 = LegacyResidualBlock(c3)
        self.res4 = LegacyResidualBlock(c3)
        self.res5 = LegacyResidualBlock(c3)
        self.deconv1 = LegacyConvLayer(c3, c2, 3, 1)
        self.in4 = nn.InstanceNorm2d(c2, affine=True)
        self.deconv2 = LegacyConvLayer(c2, c1, 3, 1)
        self.in5 = nn.InstanceNorm2d(c1, affine=True)
        self.deconv3 = LegacyConvLayer(c1, 3, 9, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.in1(self.conv1(x)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))
        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.relu(self.in4(self.deconv1(nn.functional.interpolate(y, scale_factor=2, mode="nearest"))))
        y = self.relu(self.in5(self.deconv2(nn.functional.interpolate(y, scale_factor=2, mode="nearest"))))
        return self.deconv3(y)


def load_any_teacher(path, default_width=0.25):
    """Load any teacher model checkpoint (Ti60 BN, Stream, or legacy InstanceNorm)."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "metadata" in raw:
        meta = raw["metadata"]
        arch = meta.get("architecture")
        if arch == STREAM_ARCHITECTURE:
            return load_stream_checkpoint(path)
        elif arch == ARCHITECTURE:
            model = TransformerNet(meta.get("width", default_width))
            model.load_state_dict(raw["state_dict"], strict=True)
            return model.eval(), meta
        else:
            raise ValueError(f"Unknown architecture in metadata: {arch}")
    elif isinstance(raw, dict):
        # Legacy InstanceNorm state dict
        first_conv = raw.get("conv1.conv2d.weight")
        c1 = first_conv.shape[0] if first_conv is not None else int(32 * default_width)
        w = c1 / 32.0
        model = LegacyTransformerNet(width=w)
        model.load_state_dict(raw, strict=True)
        return model.eval(), {"architecture": "legacy_instancenorm_v0", "width": w, "image_size": 256}
    else:
        raise ValueError(f"Unrecognized checkpoint object type: {type(raw)}")

