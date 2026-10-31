#!/usr/bin/env python3
"""Knowledge Distillation and INT8 Hardware Exporter for Ti60 Stream CNN.

Distills a heavy style transfer model (e.g. TransformerNet, 16 layers, 104k params)
into the Ti60 real-time on-chip line-buffer Student CNN (3 layers, 316 params, 0 DDR intermediate traffic).
Outputs bit-exact C parameters (stream_model.h) ready for FPGA firmware deployment at 720p60.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local imports work whether run from repo root or package dir
REPO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_DIR))

from neural_style.stream_net import (
    STREAM_ARCHITECTURE,
    StreamConv,
    StreamStudentNet,
    load_stream_checkpoint,
    save_stream_checkpoint,
)
from neural_style.transformer_net import load_any_teacher


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill Teacher Style Model to Ti60 Line-Buffer Student CNN (Scheme C)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Model & Teacher
    parser.add_argument("--teacher", default=None,
                        help="Path to teacher model checkpoint (.model from neural_style.py or legacy model)")
    parser.add_argument("--teacher-width", type=float, default=0.25,
                        help="Channel width multiplier for legacy teacher models (0.25 or 1.0)")
    parser.add_argument("--style-image", default="images/style-images/monet.jpg",
                        help="Path to style image (used for auxiliary style loss or direct training)")
    parser.add_argument("--candidate", default="micro4_d2", choices=["micro4_d1", "micro4_d2", "micro4_d4"],
                        help="Hardware student candidate (dilation rate for layer 1)")
    parser.add_argument("--dilation", type=int, default=2,
                        help="Dilation factor for layer 1 (matches FPGA RTL iris_stream_conv DILATION parameter)")

    # Data
    parser.add_argument("--dataset", default="images/calib",
                        help="Directory containing training/content images (calib, COCO, etc.)")
    parser.add_argument("--image-size", type=int, default=192,
                        help="Crop patch size for training (must be multiple of 4)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for distillation training")
    parser.add_argument("--calib-count", type=int, default=50,
                        help="Number of images to use for INT8 activation calibration")

    # Optimization
    parser.add_argument("--steps", type=int, default=6000,
                        help="Total distillation optimization steps")
    parser.add_argument("--lr", type=float, default=7e-4,
                        help="Initial learning rate")
    parser.add_argument("--lr-decay-step", type=int, default=4200,
                        help="Step to decay learning rate to fine-tuning rate (2e-4)")
    parser.add_argument("--grad-weight", type=float, default=0.6,
                        help="Weight for spatial gradient difference loss (sharpens edges & text contours)")
    parser.add_argument("--contour-weight", type=float, default=3.0,
                        help="Weight multiplier for dark lines and high-contrast facial contours")
    parser.add_argument("--style-weight", type=float, default=0.0,
                        help="Weight for optional auxiliary VGG style loss (0.0 to disable)")

    # Execution & Export
    parser.add_argument("--device", default=None,
                        help="Compute device ('cuda' or 'cpu'; auto-detects GPU by default)")
    parser.add_argument("--save-dir", default="outputs/distill_stream",
                        help="Directory to save checkpoints, previews, and exported headers")
    parser.add_argument("--export-header", default=None,
                        help="Direct output path for stream_model.h (e.g. firmware/stream_style/stream_model.h)")
    parser.add_argument("--preview-image", default=None,
                        help="Path to image for periodic visual comparison (e.g. test photo)")
    parser.add_argument("--preview-interval", type=int, default=500,
                        help="Save visual preview comparison every N steps")
    parser.add_argument("--checkpoint-interval", type=int, default=1000,
                        help="Save student checkpoint every N steps")
    parser.add_argument("--resume", default=None,
                        help="Path to student checkpoint (.pt) to resume training from")
    return parser.parse_args()


def find_images(dataset_dir):
    """Recursively collect image paths."""
    p = Path(dataset_dir)
    if not p.exists():
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' does not exist.")
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    images = []
    for ext in exts:
        images.extend(p.glob(f"**/{ext}"))
        images.extend(p.glob(f"**/{ext.upper()}"))
    images = sorted(list(set(images)))
    if not images:
        raise ValueError(f"No valid images found in '{dataset_dir}'.")
    return images


def load_rgb_image(path, target_size=None):
    """Load image as RGB float tensor [3, H, W] in 0..255."""
    with Image.open(path) as img:
        im = img.convert("RGB")
        if target_size is not None:
            w, h = im.size
            if w < h:
                new_w, new_h = target_size, int(round(h * target_size / w))
            else:
                new_w, new_h = int(round(w * target_size / h)), target_size
            im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)
            left = (new_w - target_size) // 2
            top = (new_h - target_size) // 2
            im = im.crop((left, top, left + target_size, top + target_size))
        arr = np.array(im, dtype=np.float32).transpose(2, 0, 1)
        return torch.from_numpy(arr)


def quantize_multiplier(scale):
    """Decompose real scale into (multiplier, shift) per TFLite / gemmlowp standard."""
    sig, exp = math.frexp(scale)
    fixed = int(math.floor(sig * (1 << 31) + 0.5))
    if fixed == (1 << 31):
        fixed //= 2
        exp += 1
    if exp < -31:
        fixed, exp = 0, 0
    if not (-31 <= exp <= 0):
        raise ValueError(f"Hardware requires shift in [-31, 0], got {exp} for scale {scale}")
    return fixed, exp


def run_int8_conv_reference(x, w, bias, mult, shift, iz, oz, s, d, k):
    """Exact bit-level simulation of Ti60 iris_stream_conv RTL arithmetic."""
    h, w_in = x.shape[:2]
    oh, ow = (h + s - 1) // s, (w_in + s - 1) // s
    ph = max(0, (oh - 1) * s + (k - 1) * d + 1 - h)
    pw = max(0, (ow - 1) * s + (k - 1) * d + 1 - w_in)
    centered = np.pad(x.astype(np.int64) - iz, ((ph // 2, ph - ph // 2), (pw // 2, pw - pw // 2), (0, 0)))
    acc = np.broadcast_to(bias, (oh, ow, 4)).astype(np.int64).copy()
    for ky in range(k):
        for kx in range(k):
            tile = centered[ky * d : ky * d + oh * s : s, kx * d : kx * d + ow * s : s]
            acc += np.einsum("hwc,oc->hwo", tile, w[:, ky, kx, :].astype(np.int64))
    high = (acc * mult + (1 << 30)) >> 31
    rs = -shift
    mask = (1 << rs) - 1
    rounded = (high >> rs) + ((high & mask) > ((mask >> 1) + (high < 0)))
    return np.clip(rounded + oz, oz, 127).astype(np.int8)


def export_hardware_header(student_model, calib_images, output_header_path, preview_image_path=None, save_dir=None):
    """Calibrate activations, quantize to INT8, and emit stream_model.h."""
    print("\n--- Starting Hardware INT8 Calibration & Export ---", flush=True)
    header_path = Path(output_header_path)
    header_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Fold BatchNorm into Conv layers
    fused_layers = []
    for i, layer in enumerate(student_model.layers):
        w, b = layer.folded()
        if i == 0:
            # First layer input: pad 3ch RGB to 4ch RGBA with zeros
            # w shape: [kh, kw, cin=3, cout=4] -> pad cin to 4
            w = np.pad(w / 255.0, ((0, 0), (0, 0), (0, 1), (0, 0)))
        elif i == 2:
            # Third layer output: scale back to 0..255 pixel range and pad cout to 4
            # w shape: [kh=1, kw=1, cin=4, cout=3] -> pad cout to 4
            w = np.pad(w * 255.0, ((0, 0), (0, 0), (0, 0), (0, 1)))
            b = np.pad(b * 255.0, (0, 1))
        # Transpose w to [cout, kh, kw, cin] for easy slicing
        w_t = w.transpose(3, 0, 1, 2)
        fused_layers.append({
            "w": w_t,
            "b": b,
            "kernel": layer.kernel,
            "stride": layer.stride,
            "dilation": layer.dilation,
        })

    # 2. Calibration: run calibration images to record dynamic range
    print(f"Calibrating activation scales using {len(calib_images)} images...", flush=True)
    max_acts = [1.0, 1.0, 255.0]  # default fallback
    calib_tensors = []
    for p in calib_images[:50]:
        img = load_rgb_image(p, target_size=256)
        calib_tensors.append(img)

    if calib_tensors:
        batch_x = torch.stack(calib_tensors).to("cpu")
        with torch.no_grad():
            fused_student = student_model.fused().to("cpu")
            x = batch_x / 255.0
            a0 = fused_student.layers[0](x)
            a1 = fused_student.layers[1](a0)
            a2 = fused_student.layers[2](a1)

            max_acts[0] = float(torch.clamp(a0.max(), min=1e-3))
            max_acts[1] = float(torch.clamp(a1.max(), min=1e-3))
            max_acts[2] = 255.0  # Final layer is bounded by RGB range 0..255

    print(f"Calibrated activation max values: layer0={max_acts[0]:.4f}, layer1={max_acts[1]:.4f}, layer2={max_acts[2]:.4f}")

    # 3. Quantization setup
    # Hardware contract:
    # iz = -128, oz = -128 for all layers.
    # input scale = 1.0, output scale = 1.0 for final layer.
    scales_act = [
        1.0,                       # input scale
        max_acts[0] / 255.0,       # layer 0 output scale
        max_acts[1] / 255.0,       # layer 1 output scale
        1.0,                       # layer 2 output scale (RGB 0..255)
    ]

    layer_parameters = []
    quantized_layers = []

    for idx, lyr in enumerate(fused_layers):
        w_float = lyr["w"]  # shape: [cout=4, kh, kw, cin=4]
        b_float = lyr["b"]  # shape: [cout=4]
        s_in = scales_act[idx]
        s_out = scales_act[idx + 1]

        # Weight quantization: symmetric per-channel
        w_scales = np.maximum(np.max(np.abs(w_float), axis=(1, 2, 3)) / 127.0, 1e-7)
        w_int8 = np.clip(np.round(w_float / w_scales[:, None, None, None]), -127, 127).astype(np.int8)

        # Bias quantization: int32
        b_int32 = np.round(b_float / (s_in * w_scales)).astype(np.int32)

        # Multiplier and shift calculation
        mults = []
        shifts = []
        for oc in range(4):
            m_real = float(s_in * w_scales[oc] / s_out)
            m_fixed, exp = quantize_multiplier(m_real)
            mults.append(m_fixed)
            shifts.append(exp)

        # Pack into hardware tap format: 9 taps x 4 words = 36 words
        # w_int8 shape: [4, kh, kw, 4].
        taps = np.zeros((9, 4, 4), dtype=np.int8)
        kh, kw = lyr["kernel"], lyr["kernel"]
        for ky in range(kh):
            for kx in range(kw):
                taps[ky * kw + kx] = w_int8[:, ky, kx, :]

        params = []
        for tap in taps:
            raw = tap.tobytes()
            params.extend(int.from_bytes(raw[j : j + 4], "little") for j in range(0, 16, 4))

        iz = -128
        oz = -128
        params.extend([int(x) & 0xFFFFFFFF for x in b_int32])
        params.extend(mults)
        params.extend([int(x) & 0xFFFFFFFF for x in shifts])
        params.extend([iz & 0xFF, oz & 0xFF, oz & 0xFF, 127])

        if len(params) != 52:
            raise ValueError(f"Expected exactly 52 parameters for layer {idx}, got {len(params)}")

        layer_parameters.append(params)
        quantized_layers.append({
            "w": w_int8,
            "b": b_int32,
            "mult": np.array(mults, dtype=np.int64),
            "shift": np.array(shifts, dtype=np.int64),
            "iz": iz,
            "oz": oz,
            "s": lyr["stride"],
            "d": lyr["dilation"],
            "k": lyr["kernel"],
        })

    # 4. Generate C Header
    header_content = (
        "#pragma once\n"
        "#include <stdint.h>\n"
        "// Automatically generated by fast_neural_style/distill_stream.py\n"
        "// Architecture: Ti60 3-layer line buffer CNN (Conv3s2 -> Conv3d2 -> Conv1s1 -> NN2x)\n"
        "// Parameters: 3 layers x 52 uint32 words (weights, bias, multipliers, shifts, activation limits)\n"
        "static const uint32_t iris_cnn_parameters[3][52] = {\n"
        + ",\n".join("    {" + ", ".join(f"0x{v:08x}u" for v in lyr) + "}" for lyr in layer_parameters)
        + "\n};\n"
    )
    header_path.write_text(header_content)
    print(f"[SUCCESS] Exported hardware C header to: {header_path}", flush=True)

    # 5. Run hardware exact bit-level simulation on preview image
    if preview_image_path and Path(preview_image_path).exists():
        print(f"Running hardware bit-accurate verification on preview: {preview_image_path}...", flush=True)
        img_rgb = load_rgb_image(preview_image_path)
        h, w = img_rgb.shape[1], img_rgb.shape[2]
        # Align to even dimensions
        h_even = (h // 2) * 2
        w_even = (w // 2) * 2
        img_np = img_rgb[:, :h_even, :w_even].numpy().transpose(1, 2, 0)

        # Pad 4th channel
        inp_rgba = np.pad(img_np, ((0, 0), (0, 0), (0, 1))) - 128
        inp_int8 = np.clip(inp_rgba, -128, 127).astype(np.int8)

        # Layer 0
        l0 = quantized_layers[0]
        y0 = run_int8_conv_reference(inp_int8, l0["w"], l0["b"], l0["mult"], l0["shift"], l0["iz"], l0["oz"], l0["s"], l0["d"], l0["k"])
        # Layer 1
        l1 = quantized_layers[1]
        y1 = run_int8_conv_reference(y0, l1["w"], l1["b"], l1["mult"], l1["shift"], l1["iz"], l1["oz"], l1["s"], l1["d"], l1["k"])
        # Layer 2
        l2 = quantized_layers[2]
        y2 = run_int8_conv_reference(y1, l2["w"], l2["b"], l2["mult"], l2["shift"], l2["iz"], l2["oz"], l2["s"], l2["d"], l2["k"])

        # Convert back to float [0..255] and upsample nearest 2x
        rgb_low = (y2[..., :3].astype(np.float32) + 128.0)
        rgb_hd = np.repeat(np.repeat(rgb_low, 2, axis=0), 2, axis=1)
        rgb_hd = np.clip(rgb_hd, 0, 255).astype(np.uint8)

        # Save verification output image
        out_prev = (Path(save_dir) if save_dir else header_path.parent) / "hardware_verified_preview.png"
        Image.fromarray(rgb_hd).save(out_prev)
        print(f"[SUCCESS] Saved hardware bit-accurate verification image: {out_prev}", flush=True)


def train_distillation(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}", flush=True)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Dataset
    images = find_images(args.dataset)
    print(f"Found {len(images)} content images in '{args.dataset}'.", flush=True)

    # 2. Setup Student Model
    candidate_dilation = {"micro4_d1": 1, "micro4_d2": 2, "micro4_d4": 4}.get(args.candidate, args.dilation)
    student = StreamStudentNet(dilation=candidate_dilation).to(device)
    print(f"Initialized StreamStudentNet (candidate={args.candidate}, dilation={candidate_dilation}).", flush=True)
    param_count = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count} (Ti60 on-chip line-buffer stream, 0 DDR intermediate traffic)", flush=True)

    # Resume if requested
    start_step = 0
    if args.resume and Path(args.resume).exists():
        print(f"Resuming student from {args.resume}...", flush=True)
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        student.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)
        start_step = ckpt.get("step", 0) + 1

    # 3. Setup Teacher Model
    teacher = None
    if args.teacher and Path(args.teacher).exists():
        print(f"Loading teacher model from '{args.teacher}'...", flush=True)
        teacher, meta = load_any_teacher(args.teacher, default_width=args.teacher_width)
        teacher = teacher.to(device).eval()
        print(f"Teacher loaded successfully: {meta}", flush=True)
    else:
        print("No teacher checkpoint specified; using direct style loss distillation mode.", flush=True)

    # Setup VGG if style loss is enabled or if no teacher is provided
    vgg = None
    style_grams = []
    if (args.style_weight > 0 or teacher is None) and args.style_image and Path(args.style_image).exists():
        print(f"Loading VGG16 and computing style representation for '{args.style_image}'...", flush=True)
        from neural_style.vgg import Vgg16
        import neural_style.utils as utils
        vgg = Vgg16(requires_grad=False).to(device).eval()
        style_img = load_rgb_image(args.style_image, target_size=256).unsqueeze(0).to(device)
        with torch.no_grad():
            style_feats = vgg(utils.normalize_batch(style_img))
            style_grams = [utils.gram_matrix(f) for f in style_feats]
        print("Style representations calculated.", flush=True)

    # 4. Optimization Setup
    optimizer = torch.optim.Adam(student.parameters(), lr=args.lr)
    l1_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()

    preview_img_path = args.preview_image
    if not preview_img_path:
        # Use first image as preview fallback
        preview_img_path = str(images[0])

    print("\n--- Starting Distillation Training ---", flush=True)
    start_time = time.time()
    student.train()

    for step in range(start_step, args.steps):
        # Scheduled Learning Rate
        lr = args.lr if step < args.lr_decay_step else 2e-4
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Sample random crops for batch
        batch_crops = []
        for _ in range(args.batch_size):
            img_path = random.choice(images)
            img = load_rgb_image(img_path)  # [3, H, W]
            h, w = img.shape[1], img.shape[2]
            sz = args.image_size
            if h < sz or w < sz:
                # Resize if image is smaller than crop size
                scale = max(sz / h, sz / w)
                nh, nw = int(math.ceil(h * scale)), int(math.ceil(w * scale))
                img = F.interpolate(img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False).squeeze(0)
                h, w = nh, nw
            # Align crop coordinates to multiples of 4 for consistent sub-sampling
            top = (random.randrange(0, h - sz + 1) // 4) * 4
            left = (random.randrange(0, w - sz + 1) // 4) * 4
            crop = img[:, top : top + sz, left : left + sz]
            if random.random() < 0.5:
                crop = torch.flip(crop, dims=[-1])
            batch_crops.append(crop)

        x_batch = torch.stack(batch_crops).to(device)  # [B, 3, sz, sz], 0..255

        # Forward passes
        optimizer.zero_grad(set_to_none=True)
        pred_y = student(x_batch)  # [B, 3, sz, sz], 0..255

        total_loss = torch.tensor(0.0, device=device)
        log_dict = {}

        if teacher is not None:
            with torch.no_grad():
                target_y = teacher(x_batch).clamp(0, 255)

            # Avoid margin boundary effects (8px interior margin)
            p_inner = pred_y[:, :, 8:-8, 8:-8] / 255.0
            t_inner = target_y[:, :, 8:-8, 8:-8] / 255.0

            # Contour / dark-line weighting: higher attention on sharp black lines & facial contours
            contour_w = 1.0 + args.contour_weight * (1.0 - t_inner.mean(dim=1, keepdim=True))
            distill_loss = (torch.abs(p_inner - t_inner) * contour_w).mean()
            total_loss = total_loss + distill_loss
            log_dict["distill"] = distill_loss.item()

            # Gradient difference loss (Sobel-like spatial derivatives along X and Y)
            if args.grad_weight > 0:
                grad_dx = F.l1_loss(torch.diff(p_inner, dim=-1), torch.diff(t_inner, dim=-1))
                grad_dy = F.l1_loss(torch.diff(p_inner, dim=-2), torch.diff(t_inner, dim=-2))
                grad_loss = args.grad_weight * (grad_dx + grad_dy)
                total_loss = total_loss + grad_loss
                log_dict["grad"] = grad_loss.item()
        else:
            # Direct perceptual loss mode
            import neural_style.utils as utils
            pred_feats = vgg(utils.normalize_batch(pred_y))
            with torch.no_grad():
                content_feats = vgg(utils.normalize_batch(x_batch))
            c_loss = 1e5 * mse_loss_fn(pred_feats.relu2_2, content_feats.relu2_2)
            s_loss = (args.style_weight or 1e10) * sum(
                mse_loss_fn(utils.gram_matrix(f), target.expand(x_batch.shape[0], -1, -1))
                for f, target in zip(pred_feats, style_grams)
            )
            total_loss = total_loss + c_loss + s_loss
            log_dict["content"] = c_loss.item()
            log_dict["style"] = s_loss.item()

        if vgg is not None and args.style_weight > 0 and teacher is not None:
            import neural_style.utils as utils
            pred_feats = vgg(utils.normalize_batch(pred_y))
            aux_style_loss = args.style_weight * sum(
                mse_loss_fn(utils.gram_matrix(f), target.expand(x_batch.shape[0], -1, -1))
                for f, target in zip(pred_feats, style_grams)
            )
            total_loss = total_loss + aux_style_loss
            log_dict["aux_style"] = aux_style_loss.item()

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        # Logging
        if (step + 1) % 100 == 0 or step == start_step:
            elapsed = time.time() - start_time
            loss_str = " ".join(f"{k}={v:.4f}" for k, v in log_dict.items())
            print(f"Step [{step + 1}/{args.steps}] ({elapsed:.1f}s) lr={lr:.1e} total_loss={total_loss.item():.4f} | {loss_str}", flush=True)

        # Periodic visual preview
        if ((step + 1) % args.preview_interval == 0 or step + 1 == args.steps) and preview_img_path:
            with torch.no_grad():
                student.eval()
                prev_x = load_rgb_image(preview_img_path).unsqueeze(0).to(device)
                h_p, w_p = (prev_x.shape[2] // 4) * 4, (prev_x.shape[3] // 4) * 4
                prev_x = prev_x[:, :, :h_p, :w_p]

                prev_student = student(prev_x).clamp(0, 255).cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                panels = [prev_x.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)]

                if teacher is not None:
                    prev_teacher = teacher(prev_x).clamp(0, 255).cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
                    panels.append(prev_teacher)

                panels.append(prev_student)
                combined = np.concatenate(panels, axis=1)
                preview_file = save_dir / f"preview_step_{step + 1}.png"
                Image.fromarray(combined).save(preview_file)
                print(f"[Preview saved] {preview_file}", flush=True)
                student.train()

        # Periodic Checkpoint
        if (step + 1) % args.checkpoint_interval == 0 or step + 1 == args.steps:
            ckpt_path = save_dir / f"student_{args.candidate}_step_{step + 1}.pt"
            save_stream_checkpoint(ckpt_path, student, args.image_size, step=step, total_loss=total_loss.item())
            print(f"[Checkpoint saved] {ckpt_path}", flush=True)

    # 5. Export Final Checkpoint and Ti60 stream_model.h Header
    final_ckpt = save_dir / f"student_{args.candidate}_final.pt"
    save_stream_checkpoint(final_ckpt, student, args.image_size, step=args.steps, total_loss=total_loss.item())

    export_header_path = args.export_header if args.export_header else (save_dir / "stream_model.h")
    export_hardware_header(
        student_model=student,
        calib_images=images,
        output_header_path=export_header_path,
        preview_image_path=preview_img_path,
        save_dir=save_dir,
    )
    print("\n[SUCCESS] Distillation training and Ti60 hardware export complete!", flush=True)
    print(f"Hardware header: {export_header_path}")
    print(f"Checkpoints and previews: {save_dir}")


def main():
    args = parse_args()
    train_distillation(args)


if __name__ == "__main__":
    main()
