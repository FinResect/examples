import copy
import math

import torch
from torch import nn


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
    if metadata.get("architecture") != ARCHITECTURE:
        raise ValueError("Unsupported checkpoint architecture")
    size = metadata.get("image_size")
    if not isinstance(size, int) or size < 16 or size % 4:
        raise ValueError("Checkpoint image_size must be >=16 and divisible by four")
    model = TransformerNet(metadata["width"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), metadata
