import copy
import math

import torch
from torch import nn


ARCHITECTURE = "ti60_style_bn_3x3_v1"
DW_ARCHITECTURE = "ti60_style_dw_c4_s_v1"


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, normalize=True,
                 kernel_size=3, groups=1):
        super().__init__()
        # For even inputs, TF SAME stride=2 pads only the right and bottom.
        self.pad = nn.ZeroPad2d((0, 1, 0, 1)) if stride == 2 else nn.Identity()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding=kernel_size // 2 if stride == 1 else 0,
                              groups=groups)
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
    architecture = ARCHITECTURE

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


class DWTransformerNet(nn.Module):
    """DW-C4-S: 7 depthwise, 9 pointwise, 2 resizes; 12,979,200 VGA MACs."""

    architecture = DW_ARCHITECTURE

    def __init__(self):
        super().__init__()
        layers = []
        # Two downsamplers, four spatial groups, then one compression group.
        for channels, stride in ((3, 2), (4, 2)) + ((4, 1),) * 5:
            layers.extend([
                ConvLayer(channels, channels, stride=stride, groups=channels),
                nn.ReLU(),
                ConvLayer(channels, 4, kernel_size=1),
                nn.ReLU(),
            ])
        layers.extend([
            ConvLayer(4, 4, kernel_size=1), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvLayer(4, 3, kernel_size=1, normalize=False),
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

    def fused(self):
        return TransformerNet.fused(self)


def create_model(architecture="dw_c4_s", width=None):
    """Build from a CLI name or checkpoint architecture ID; width is BN-only."""
    if architecture in ("dw_c4_s", DW_ARCHITECTURE):
        if width is not None:
            raise ValueError("--width is only supported for bn_3x3")
        return DWTransformerNet()
    if architecture in ("bn_3x3", ARCHITECTURE):
        return TransformerNet(0.25 if width is None else width)
    raise ValueError(f"Unsupported architecture: {architecture}")


def checkpoint_dimensions(metadata):
    """Return validated (height, width), including shipped square BN checkpoints."""
    if metadata.get("architecture") not in (ARCHITECTURE, DW_ARCHITECTURE):
        raise ValueError("Unsupported checkpoint architecture")
    if "image_height" in metadata or "image_width" in metadata:
        height, width = metadata.get("image_height"), metadata.get("image_width")
        size = metadata.get("image_size")
        expected = height if height == width else (height, width)
        if size != expected:
            raise ValueError("Checkpoint image_size disagrees with image_height/image_width")
    else:
        height = width = metadata.get("image_size")
    if any(type(value) is not int or value < 16 or value % 4
           for value in (height, width)):
        raise ValueError("Checkpoint dimensions must be >=16 and divisible by four")
    if metadata["architecture"] == DW_ARCHITECTURE and "image_height" not in metadata:
        raise ValueError("DW checkpoint requires image_height and image_width")
    return height, width


def save_checkpoint(path, model, image_size, **extra):
    """Save unfused model weights; image_size accepts an int or (height, width)."""
    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    metadata = {"architecture": model.architecture,
                "image_size": height if height == width else (height, width)}
    if model.architecture == ARCHITECTURE:
        metadata["width"] = model.width
    if model.architecture == DW_ARCHITECTURE or height != width:
        metadata.update(image_height=height, image_width=width)
    checkpoint_dimensions(metadata)
    torch.save({
        **extra,
        "metadata": metadata,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }, path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "metadata" not in checkpoint:
        raise ValueError("Legacy InstanceNorm/raw state_dict models are incompatible; retrain the Ti60 model")
    metadata = checkpoint["metadata"]
    checkpoint_dimensions(metadata)
    if metadata["architecture"] == ARCHITECTURE and "width" not in metadata:
        raise ValueError("BN checkpoint requires width metadata")
    model = create_model(metadata["architecture"], metadata.get("width"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), metadata
