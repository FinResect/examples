"""Stream Student Network for Ti60 real-time line-buffer inference.

Hardware pipeline: Conv3s2 -> Conv3d2 -> Conv1s1 -> NN2x DDR write back.
Zero intermediate DDR read/write; executes entirely within FPGA on-chip line buffers.
"""
import copy
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

STREAM_ARCHITECTURE = "ti60_stream_c444_v1"


class StreamConv(nn.Module):
    """Convolution layer matching Ti60 iris_stream_conv line-buffer hardware.

    Pads input with TF 'SAME' convention so boundary dimensions align exactly with hardware.
    Includes optional BatchNorm for stable training, which folds cleanly into conv weights for deployment.
    """
    def __init__(self, in_channels, out_channels, kernel=3, stride=1, dilation=1, activation=True, bn=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride=stride, dilation=dilation, bias=True)
        self.bn = nn.BatchNorm2d(out_channels) if bn else nn.Identity()

    def forward(self, x):
        h, w = x.shape[-2:]
        k = (self.kernel - 1) * self.dilation + 1
        ph = max(0, ((h + self.stride - 1) // self.stride - 1) * self.stride + k - h)
        pw = max(0, ((w + self.stride - 1) // self.stride - 1) * self.stride + k - w)
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        x = F.pad(x, pad)
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x) if self.activation else x

    def folded(self):
        """Fold BatchNorm into convolution weights and bias.

        Returns:
            w: numpy array of shape (kernel_h, kernel_w, in_channels, out_channels)
            b: numpy array of shape (out_channels,)
        """
        w = self.conv.weight.detach().clone()
        b = self.conv.bias.detach().clone() if self.conv.bias is not None else torch.zeros(self.out_channels, device=w.device)
        if isinstance(self.bn, nn.BatchNorm2d):
            scale = self.bn.weight.detach() / torch.sqrt(self.bn.running_var.detach() + self.bn.eps)
            w = w * scale[:, None, None, None]
            b = (b - self.bn.running_mean.detach()) * scale + self.bn.bias.detach()
        # Transpose from PyTorch [cout, cin, kh, kw] to TF/TFLite [kh, kw, cin, cout]
        w_np = w.cpu().numpy().transpose(2, 3, 1, 0)
        b_np = b.cpu().numpy()
        return w_np, b_np


class StreamStudentNet(nn.Module):
    """Ti60 3-layer line buffer student network.

    Layer 0: Conv3s2 (3 -> 4 channels)
    Layer 1: Conv3d2 (4 -> 4 channels, dilation=2)
    Layer 2: Conv1s1 (4 -> 3 channels)
    Output:  NN 2x upsampling

    Trainable weights: 264 weights + 11 biases (0 DDR intermediate traffic).
    Maps to Ti60 iris_stream_cnn 4x4 hardware pipeline via channel padding during export.
    """
    def __init__(self, dilation=2, in_channels=3, out_channels=3):
        super().__init__()
        self.dilation = dilation
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = nn.ModuleList([
            StreamConv(in_channels, 4, kernel=3, stride=2, dilation=1, activation=True, bn=True),
            StreamConv(4, 4, kernel=3, stride=1, dilation=dilation, activation=True, bn=True),
            StreamConv(4, out_channels, kernel=1, stride=1, dilation=1, activation=True, bn=False),
        ])

    def forward(self, x):
        """Forward pass.

        Accepts:
            x: Tensor of shape [B, 3, H, W] (RGB), pixel values in [0, 255] or [0, 1].
        Returns:
            Styled tensor of same scale and spatial resolution, [B, 3, H, W].
        """
        orig_shape = x.shape
        is_normalized = (x.max() <= 1.05 and x.min() >= -0.05) if x.numel() > 0 else False
        feat = x if is_normalized else (x / 255.0)

        for layer in self.layers:
            feat = layer(feat)

        # Nearest neighbor 2x upsample to restore input resolution
        out = F.interpolate(feat, size=orig_shape[-2:], mode="nearest")

        if not is_normalized:
            out = out * 255.0
        return out

    def fused(self):
        """Return a copy of the network with all BatchNorm layers fused into Conv."""
        model = copy.deepcopy(self).eval()
        for layer in model.layers:
            if isinstance(layer, StreamConv) and isinstance(layer.bn, nn.BatchNorm2d):
                layer.conv = nn.utils.fusion.fuse_conv_bn_eval(layer.conv, layer.bn)
                layer.bn = nn.Identity()
        return model.eval()


def save_stream_checkpoint(path, model, image_size, **extra):
    """Save StreamStudentNet checkpoint with metadata."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "metadata": {
            "architecture": STREAM_ARCHITECTURE,
            "dilation": model.dilation,
            "in_channels": model.in_channels,
            "out_channels": model.out_channels,
            "image_size": image_size,
        },
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        **extra,
    }, path)


def load_stream_checkpoint(path):
    """Load StreamStudentNet checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "metadata" not in checkpoint:
        raise ValueError("Invalid stream checkpoint format")
    metadata = checkpoint["metadata"]
    if metadata.get("architecture") != STREAM_ARCHITECTURE:
        raise ValueError(f"Expected architecture {STREAM_ARCHITECTURE}, got {metadata.get('architecture')}")
    model = StreamStudentNet(dilation=metadata.get("dilation", 2))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval(), metadata
