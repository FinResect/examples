import argparse
from functools import partial
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

import utils
from preprocessing import preprocess_image
from transformer_net import checkpoint_dimensions, create_model, load_checkpoint, save_checkpoint


def image_transform(size):
    return partial(preprocess_image, image_size=size)


def image_dimensions(args, default):
    if (args.image_height is None) != (args.image_width is None):
        raise ValueError("--image-height and --image-width must be supplied together")
    if args.image_size is not None and args.image_height is not None:
        raise ValueError("--image-size is mutually exclusive with --image-height/--image-width")
    size = args.image_size
    dimensions = ((args.image_height, args.image_width) if args.image_height is not None
                  else (size, size) if size is not None else default)
    if any(value < 16 or value % 4 for value in dimensions):
        raise ValueError("Image dimensions must be >=16 and divisible by four")
    return dimensions


def train(args, device):
    from vgg import Vgg16

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dimensions = image_dimensions(args, (128, 128))
    dataset = datasets.ImageFolder(args.dataset, image_transform(dimensions))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    model = create_model(args.architecture, args.width).to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    vgg = Vgg16(requires_grad=False).to(device).eval()
    style = image_transform(args.style_size or dimensions)(
        utils.load_image(args.style_image)).unsqueeze(0).to(device)
    with torch.no_grad():
        style_grams = [utils.gram_matrix(f) for f in vgg(utils.normalize_batch(style))]
    loss_fn = torch.nn.MSELoss()
    output_dir = Path(args.save_model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_model_dir) if args.checkpoint_model_dir else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        totals = np.zeros(2)
        samples = 0
        for step, (images, _) in enumerate(loader, 1):
            x = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            y = model(x)
            features_y = vgg(utils.normalize_batch(y))
            with torch.no_grad():
                features_x = vgg(utils.normalize_batch(x))
            content_loss = args.content_weight * loss_fn(features_y.relu2_2, features_x.relu2_2)
            style_loss = args.style_weight * sum(
                loss_fn(utils.gram_matrix(f), target.expand(x.shape[0], -1, -1))
                for f, target in zip(features_y, style_grams))
            loss = content_loss + style_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, batch {step}")
            loss.backward()
            optimizer.step()
            totals += np.array([content_loss.item(), style_loss.item()]) * x.shape[0]
            samples += x.shape[0]
            if step % args.log_interval == 0 or step == len(loader):
                print(f"epoch={epoch} batch={step}/{len(loader)} "
                      f"content={totals[0]/samples:.6f} style={totals[1]/samples:.6f}", flush=True)
            if checkpoint_dir and step % args.checkpoint_interval == 0:
                save_checkpoint(checkpoint_dir / f"epoch_{epoch}_batch_{step}.model",
                                model, dimensions, epoch=epoch, batch=step)
        path = output_dir / f"ti60_epoch_{epoch}_{time.strftime('%Y%m%d_%H%M%S')}.model"
        save_checkpoint(path, model, dimensions, epoch=epoch)
        print(f"Saved {path}", flush=True)


def stylize(args, device):
    model, metadata = load_checkpoint(args.model)
    if args.architecture is not None:
        expected = create_model(args.architecture, args.width).architecture
        if expected != metadata["architecture"]:
            raise ValueError("--architecture does not match checkpoint metadata")
    if args.width is not None:
        if "width" not in metadata:
            raise ValueError("--width is only supported for bn_3x3")
        if args.width != metadata["width"]:
            raise ValueError("--width does not match checkpoint metadata")
    dimensions = image_dimensions(args, checkpoint_dimensions(metadata))
    x = image_transform(dimensions)(utils.load_image(args.content_image)).unsqueeze(0)
    model = model.fused().to(device)
    with torch.inference_mode():
        output = model(x.to(device)).cpu()
    Path(args.output_image).parent.mkdir(parents=True, exist_ok=True)
    utils.save_image(args.output_image, output[0])
    if args.export_onnx:
        Path(args.export_onnx).parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(model.cpu(), x, args.export_onnx, opset_version=13,
                          input_names=["input"], output_names=["output"], dynamo=False)


def main():
    parser = argparse.ArgumentParser(description="Ti60 DW-C4-S / legacy BN style transfer")
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--dataset", required=True)
    training.add_argument("--style-image", required=True)
    training.add_argument("--save-model-dir", required=True)
    training.add_argument("--checkpoint-model-dir")
    training.add_argument("--epochs", type=int, default=2)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--style-size", type=int)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--content-weight", type=float, default=1e5)
    training.add_argument("--style-weight", type=float, default=1e10)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--log-interval", type=int, default=100)
    training.add_argument("--checkpoint-interval", type=int, default=2000)
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--model", required=True)
    evaluation.add_argument("--content-image", required=True)
    evaluation.add_argument("--output-image", required=True)
    evaluation.add_argument("--export-onnx", "--export_onnx", dest="export_onnx")
    for command in (training, evaluation):
        command.add_argument("--image-size", type=int, help="square size (train default: 128)")
        command.add_argument("--image-height", type=int)
        command.add_argument("--image-width", type=int)
        command.add_argument("--architecture", choices=("dw_c4_s", "bn_3x3"),
                             default="dw_c4_s" if command is training else None)
        command.add_argument("--width", type=float, help="channel multiplier for bn_3x3 only")
        command.add_argument("--accel", action="store_true", help="use CUDA (server training)")
    args = parser.parse_args()
    try:
        image_dimensions(args, (128, 128))
        if args.width is not None and args.architecture == "dw_c4_s":
            raise ValueError("--width is only supported for bn_3x3")
    except ValueError as error:
        parser.error(str(error))
    if args.accel and not torch.cuda.is_available():
        parser.error("--accel requested but CUDA is unavailable")
    device = torch.device("cuda" if args.accel else "cpu")
    print(f"Using device: {device}")
    if args.command == "train":
        if min(args.epochs, args.batch_size, args.log_interval, args.checkpoint_interval) < 1 or args.workers < 0:
            parser.error("epochs, batch-size and intervals must be positive; workers must be nonnegative")
        if args.style_size is not None and args.style_size < 16:
            parser.error("--style-size must be >=16")
        train(args, device)
    else:
        stylize(args, device)


if __name__ == "__main__":
    main()
