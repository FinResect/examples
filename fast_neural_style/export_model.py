#!/usr/bin/env python3
"""
通用模型导出工具: 将 PyTorch 风格化模型导出为 Ti60 硬件加速器所需的完整产物

输出产物 (保留所需核心文件):
1. <name>.model                  - PyTorch 检查点
2. <name>_int8.tflite           - 全整型 INT8 TFLite 模型
3. rain_backend_model.h         - 8x8 矩阵加速器权重重排与层配置
4. weights.bin                  - 8x8 矩阵加速器二进制权重数据
5. <name>_int8_model_data.h     - 嵌入固件的 C 头文件
6. <name>_int8_model_data.cc    - 嵌入固件的 C 数组文件
"""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="PyTorch 训练生成的 .model 文件路径")
    parser.add_argument("--name", default=None, help="模型名称标识 (默认根据模型文件名自动推导)")
    parser.add_argument("--iris-dir", default=None, help="Iris 仓库根目录 (默认自动探测)")
    parser.add_argument("--calib-dir", default="images/calib", help="量化校准图像目录 (默认 images/calib)")
    parser.add_argument("--calib-count", type=int, default=50, help="量化校准样本数量 (默认 50)")
    parser.add_argument("--output-dir", default=None, help="输出目录 (默认 output/<name>)")
    parser.add_argument("--deploy", action="store_true", default=True, help="是否自动部署至 Iris 仓库 model_input/ 目录")
    parser.add_argument("--clean", action="store_true", default=True, help="是否自动清理 saved_model/ 和 .onnx 临时转换目录")
    return parser.parse_args()


def export_c_array(tflite_bytes: bytes, name: str, out_dir: Path):
    stem = f"{name}_int8"
    header_path = out_dir / f"{stem}_model_data.h"
    cc_path = out_dir / f"{stem}_model_data.cc"
    var_name = f"{stem}_model_data"
    len_name = f"{stem}_model_data_len"

    header_content = f"""#ifndef _{stem.upper()}_MODEL_DATA_H
#define _{stem.upper()}_MODEL_DATA_H
extern const unsigned int {len_name};
extern const unsigned char {var_name}[];
#endif
"""
    header_path.write_text(header_content)

    lines = [f'#include "{stem}_model_data.h"\n\n']
    lines.append(f"const unsigned int {len_name} = {len(tflite_bytes)};\n")
    lines.append(f"const unsigned char {var_name}[] = {{\n")
    for i in range(0, len(tflite_bytes), 12):
        chunk = tflite_bytes[i:i + 12]
        hex_vals = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {hex_vals},\n")
    lines.append("};\n")
    cc_path.write_text("".join(lines))
    print(f"   [C Data] 生成 C 数组: {header_path.name}, {cc_path.name} ({len(tflite_bytes)} 字节)")
    return header_path, cc_path


def main():
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    model_path = Path(args.model).resolve()

    if not model_path.is_file():
        sys.exit(f"❌ 找不到模型文件: {model_path}")

    # 推导模型名称
    model_name = args.name
    if not model_name:
        stem = model_path.stem
        # 去掉常见后缀如 _epoch_2, _int8 等
        for suffix in ("_epoch_1", "_epoch_2", "_epoch_3", "_epoch_4", "_epoch_5"):
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
        model_name = stem

    # 定位 Iris 目录
    if args.iris_dir:
        iris_dir = Path(args.iris_dir).resolve()
    else:
        candidates = [
            repo_dir.parent / "Iris",
            Path("/home/fin-resect/Program/FPGA/Iris")
        ]
        iris_dir = next((p for p in candidates if p.is_dir()), None)

    if not iris_dir:
        sys.exit("❌ 未定位到 Iris 仓库目录，请使用 --iris-dir 指定")

    compile_rain_script = iris_dir / ".script" / "compile-rain-backend.py"
    if not compile_rain_script.is_file():
        sys.exit(f"❌ 找不到 compile-rain-backend.py: {compile_rain_script}")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else (repo_dir / "output" / model_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f"🚀 开始导出 Ti60 8x8 加速器模型: {model_name}")
    print(f"   输入模型: {model_path}")
    print(f"   输出目录: {out_dir}")
    print(f"   Iris 目录: {iris_dir}")
    print("=" * 65)

    # 确定转换 Python 环境
    convert_py = repo_dir / ".venv-convert" / "bin" / "python"
    if not convert_py.is_file():
        convert_py = Path(sys.executable)

    # 1. 运行 model2tf_lite.py
    onnx_path = out_dir / f"{model_name}.onnx"
    saved_model_dir = out_dir / "saved_model"
    tflite_path = out_dir / f"{model_name}_int8.tflite"
    report_path = out_dir / f"{model_name}.report.json"
    calib_dir = (repo_dir / args.calib_dir).resolve()

    print("\n==> [1/3] 量化转换: PyTorch -> ONNX -> INT8 TFLite...")
    cmd_convert = [
        str(convert_py),
        str(repo_dir / "script" / "model2tf_lite.py"),
        "--model", str(model_path),
        "--width", "0.25",
        "--size", "128",
        "--onnx", str(onnx_path),
        "--saved-model", str(saved_model_dir),
        "--int8-tflite", str(tflite_path),
        "--report", str(report_path),
        "--calib", str(calib_dir),
        "--calib-count", str(args.calib_count)
    ]
    subprocess.run(cmd_convert, check=True)

    if not tflite_path.is_file():
        sys.exit(f"❌ 未能生成 TFLite 模型: {tflite_path}")
    tflite_bytes = tflite_path.read_bytes()
    print(f"   ✅ TFLite 量化完成: {tflite_path.name} ({len(tflite_bytes)} 字节)")

    # 2. 编译 8x8 矩阵加速器权重
    print("\n==> [2/3] 编译 8x8 矩阵加速器硬件权重包与参数...")
    temp_rain_dir = out_dir / "_temp_rain"
    temp_rain_dir.mkdir(parents=True, exist_ok=True)
    cmd_compile = [
        str(convert_py),
        str(compile_rain_script),
        str(tflite_path),
        str(temp_rain_dir)
    ]
    subprocess.run(cmd_compile, check=True)

    backend_header = temp_rain_dir / "rain_backend_model.h"
    backend_weights = temp_rain_dir / "weights.bin"
    if not backend_header.is_file() or not backend_weights.is_file():
        sys.exit("❌ compile-rain-backend.py 未能生成加速器产物")

    # 移动到输出根目录
    final_header = out_dir / "rain_backend_model.h"
    final_weights = out_dir / "weights.bin"
    shutil.move(str(backend_header), str(final_header))
    shutil.move(str(backend_weights), str(final_weights))
    shutil.rmtree(str(temp_rain_dir), ignore_errors=True)
    print(f"   ✅ 加速器打包完成: {final_header.name}, {final_weights.name} ({final_weights.stat().st_size} 字节)")

    # 3. 生成 C 数组
    print("\n==> [3/3] 生成固件 C 数组与整理必要产物...")
    c_header, c_source = export_c_array(tflite_bytes, model_name, out_dir)

    # 复制原始 .model 到输出目录中形成完整自包含包
    if model_path.parent != out_dir:
        shutil.copy2(model_path, out_dir / f"{model_name}.model")

    # 4. 清理中间临时文件 (saved_model, .onnx, report)
    if args.clean:
        if saved_model_dir.is_dir():
            shutil.rmtree(str(saved_model_dir), ignore_errors=True)
        if onnx_path.is_file():
            onnx_path.unlink()
        if report_path.is_file():
            report_path.unlink()
        print("   🧹 已清理 saved_model/ 和 .onnx 临时转换文件")

    # 5. 自动部署至 Iris
    if args.deploy and iris_dir:
        # 部署至 Iris/model_input (一键 build-firmware 指定输入目录)
        model_input_dir = iris_dir / "model_input"
        model_input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_header, model_input_dir / "rain_backend_model.h")
        shutil.copy2(final_weights, model_input_dir / "weights.bin")
        shutil.copy2(c_header, model_input_dir / c_header.name)
        shutil.copy2(c_source, model_input_dir / c_source.name)
        shutil.copy2(tflite_path, model_input_dir / tflite_path.name)
        print(f"\n   📦 [Deploy] 已成功部署至 Iris/model_input/:")
        print(f"      - {final_header.name}")
        print(f"      - {final_weights.name}")
        print(f"      - {c_header.name}")
        print(f"      - {c_source.name}")
        print(f"      - {tflite_path.name}")

    print("\n" + "=" * 65)
    print("🎉 导出圆满完成！输出目录中的必要产物如下:")
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"   📄 {f.name:<32} ({size_kb:8.1f} KB)")
    print("=" * 65)
    print("🚀 接下来在 Iris 目录下执行编译与烧录:")
    print("   cd ../../Iris")
    print("   .script/build-firmware")
    print("   .script/sync-iris jtag")
    print("=" * 65)


if __name__ == "__main__":
    main()
