#!/usr/bin/env python3
"""
通用风格化模型训练与一键导出脚本

网络架构: Ti60 8x8 矩阵加速器架构 (TransformerNet width=0.25, 104k 参数)
支持任意风格图片与数据集 (平铺目录或 ImageFolder 均可)。
训练完成后自动转换为全套部署产物 (INT8 TFLite, C 数组, 8x8 加速器权重头文件)，
中间临时文件自动清理，仅保留部署所需的必要产物。
"""

import argparse
from pathlib import Path
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="训练数据集目录路径 (支持平铺图片目录或标准类别子目录)")
    parser.add_argument("--style-image", required=True, help="目标风格图像文件路径 (如 images/style-images/monet.jpg)")
    parser.add_argument("--name", default=None, help="模型名称标识 (默认根据风格图片文件名推导)")
    parser.add_argument("--epochs", type=int, default=2, help="训练轮数 (默认 2)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (默认 4)")
    parser.add_argument("--image-size", type=int, default=128, help="训练图像尺寸 (默认 128)")
    parser.add_argument("--style-size", type=int, default=None, help="风格图像尺寸 (默认等于 image-size)")
    parser.add_argument("--width", type=float, default=0.25, help="通道缩放系数 (Ti60 推荐 0.25)")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率 (默认 1e-3)")
    parser.add_argument("--content-weight", type=float, default=1e5, help="内容损失权重 (默认 1e5)")
    parser.add_argument("--style-weight", type=float, default=1e10, help="风格损失权重 (默认 1e10)")
    parser.add_argument("--log-interval", type=int, default=50, help="日志打印间隔 (默认 50)")
    parser.add_argument("--workers", type=int, default=4, help="数据读取线程数 (默认 4)")
    parser.add_argument("--save-dir", default="model", help="PyTorch 模型保存目录 (默认 model/)")
    parser.add_argument("--output-dir", default=None, help="最终导出产物目录 (默认 output/<name>)")
    parser.add_argument("--calib-dir", default="images/calib", help="量化校准集目录 (默认 images/calib)")
    parser.add_argument("--no-cuda", action="store_true", help="禁用 CUDA 加速 (强制使用 CPU)")
    parser.add_argument("--export", action="store_true", default=True, help="训练完成后自动导出并部署 (默认开启)")
    parser.add_argument("--no-export", dest="export", action="store_false", help="仅训练 PyTorch 模型，不执行导出")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    style_img = Path(args.style_image).resolve()
    if not style_img.is_file():
        sys.exit(f"❌ 找不到风格图像: {style_img}")

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        sys.exit(f"❌ 找不到训练集路径: {dataset_path}")

    # 模型名称
    model_name = args.name or style_img.stem
    save_dir = (repo_dir / args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"🎨 启动神经网络风格化训练: [{model_name}]")
    print(f"   风格图像: {style_img.name}")
    print(f"   数据集:   {dataset_path}")
    print(f"   网络配置: width={args.width}, size={args.image_size}x{args.image_size}, epochs={args.epochs}")
    print(f"   硬件加速: {'CPU' if args.no_cuda else 'GPU (CUDA)'}")
    print("=" * 65)

    # 1. 运行训练
    train_cmd = [
        sys.executable,
        str(repo_dir / "neural_style" / "neural_style.py"),
        "train",
        "--dataset", str(dataset_path),
        "--style-image", str(style_img),
        "--name", model_name,
        "--save-model-dir", str(save_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--image-size", str(args.image_size),
        "--width", str(args.width),
        "--lr", str(args.lr),
        "--content-weight", str(args.content_weight),
        "--style-weight", str(args.style_weight),
        "--log-interval", str(args.log_interval),
        "--workers", str(args.workers),
    ]
    if args.style_size:
        train_cmd.extend(["--style-size", str(args.style_size)])
    if not args.no_cuda:
        train_cmd.append("--accel")

    subprocess.run(train_cmd, check=True)

    trained_model = save_dir / f"{model_name}.model"
    if not trained_model.is_file():
        # 寻找最新的 model 文件
        candidates = sorted(save_dir.glob(f"{model_name}*.model"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            trained_model = candidates[0]
        else:
            sys.exit("❌ 训练未生成模型文件！")

    print("\n" + "=" * 65)
    print(f"🎉 阶段一: PyTorch 模型训练完成！产物: {trained_model.name}")
    print("=" * 65)

    # 2. 导出与部署
    if args.export:
        print("\n==> 阶段二: 启动硬件加速器模型导出与部署...")
        export_cmd = [
            sys.executable,
            str(repo_dir / "export_model.py"),
            "--model", str(trained_model),
            "--name", model_name,
            "--calib-dir", str(args.calib_dir)
        ]
        if args.output_dir:
            export_cmd.extend(["--output-dir", str(args.output_dir)])

        subprocess.run(export_cmd, check=True)


if __name__ == "__main__":
    main()
