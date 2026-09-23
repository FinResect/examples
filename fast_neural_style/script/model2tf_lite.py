import argparse
import glob
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description=".model -> ONNX -> int8 TFLite")
    p.add_argument("--model", required=True, help="训练得到的 .model")
    p.add_argument("--width", type=float, default=0.25)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--opset", type=int, default=13)
    p.add_argument("--onnx", default="style.onnx")
    p.add_argument("--saved-model", default="saved_model")
    p.add_argument("--int8-tflite", default="style_int8.tflite")
    p.add_argument("--calib", default="calib", help="校准图目录 (jpg, 递归)")
    p.add_argument("--in-dtype", choices=["uint8", "int8"], default="uint8")
    p.add_argument("--out-dtype", choices=["uint8", "int8"], default="uint8")
    p.add_argument("--content", default=None, help="验证用内容图，留空跳过 int8 输出")
    p.add_argument("--skip-export", action="store_true")
    p.add_argument("--skip-onnx2tf", action="store_true")
    return p.parse_args()


def build_net(a):
    import torch
    from neural_style.transformer_net import TransformerNet
    net = TransformerNet(width=a.width).eval()
    sd = torch.load(a.model, map_location="cpu")
    for k in list(sd.keys()):
        if re.search(r"in\d+\.running_(mean|var)$", k):
            del sd[k]
    net.load_state_dict(sd)
    return net


def step1_export(a, net):
    import torch
    dummy = torch.randn(1, 3, a.size, a.size)
    torch.onnx.export(net, dummy, a.onnx, opset_version=a.opset,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes=None, do_constant_folding=True)
    print(f"[1/5] ONNX -> {a.onnx}")


def step2_verify_onnx(a, net):
    import numpy as np, torch, onnxruntime
    x = np.random.rand(1, 3, a.size, a.size).astype(np.float32) * 255
    with torch.no_grad():
        ref = net(torch.from_numpy(x)).numpy()
    out = onnxruntime.InferenceSession(a.onnx).run(None, {"input": x})[0]
    d = float(np.abs(ref - out).max())
    print(f"[2/5] ONNX 校验 max abs diff = {d:.6f}" + ("  (OK)" if d <= 1 else "  (偏大!)"))


def step3_onnx2tf(a):
    subprocess.run(["onnx2tf", "-i", a.onnx, "-o", a.saved_model, "-b", "1"], check=True)
    cands = glob.glob(os.path.join(a.saved_model, "*float32*.tflite"))
    path = cands[0] if cands else os.path.join(a.saved_model, "float32.tflite")
    if not cands:
        import tensorflow as tf
        open(path, "wb").write(tf.lite.TFLiteConverter.from_saved_model(a.saved_model).convert())
    print(f"[3/5] SavedModel -> {a.saved_model}; float tflite -> {path}")
    return path


def _layout(float_tflite):
    import tensorflow as tf
    it = tf.lite.Interpreter(float_tflite); it.allocate_tensors()
    shape = it.get_input_details()[0]["shape"]
    return ("nchw" if shape[1] == 3 and shape[-1] != 3 else "nhwc"), shape


def step4_int8(a, float_tflite):
    import tensorflow as tf, numpy as np
    from PIL import Image
    DT = {"uint8": tf.uint8, "int8": tf.int8}
    layout, shape = _layout(float_tflite)
    files = glob.glob(os.path.join(a.calib, "**", "*.jpg"), recursive=True)
    if not files:
        sys.exit(f"[4/5] 校准集为空: {a.calib}/**/*.jpg")
    print(f"[4/5] 校准 {len(files)} 张, 布局 {layout} {shape}")

    def rep():
        for f in files:
            arr = np.asarray(Image.open(f).convert("RGB").resize((a.size, a.size)), np.float32)
            yield [arr[None].transpose(0, 3, 1, 2) if layout == "nchw" else arr[None]]

    c = tf.lite.TFLiteConverter.from_saved_model(a.saved_model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type, c.inference_output_type = DT[a.in_dtype], DT[a.out_dtype]
    open(a.int8_tflite, "wb").write(c.convert())
    print(f"[4/5] int8 TFLite -> {a.int8_tflite}")


def step5_verify_int8(a):
    if not a.content:
        print("[5/5] 未提供 --content，跳过")
        return
    import numpy as np, tensorflow as tf
    from PIL import Image
    it = tf.lite.Interpreter(a.int8_tflite); it.allocate_tensors()
    d, o = it.get_input_details()[0], it.get_output_details()[0]
    img = np.asarray(Image.open(a.content).convert("RGB").resize((a.size, a.size)), np.uint8)
    x = img.astype(np.int16) - 128 if d["dtype"] == np.int8 else img
    it.set_tensor(d["index"], x[None].astype(d["dtype"])); it.invoke()
    y = it.get_tensor(o["index"])[0]
    if o["dtype"] == np.int8:
        y = y.astype(np.int16) + 128
    Image.fromarray(y.astype(np.uint8)).save("int8_out.jpg")
    print("[5/5] int8 输出图 -> int8_out.jpg")


def main():
    a = parse_args()
    if not a.skip_export:
        net = build_net(a)
        step1_export(a, net)
    if not os.path.exists(a.onnx):
        sys.exit(f"找不到 {a.onnx}")
    if not a.skip_export:
        step2_verify_onnx(a, net)
    if not a.skip_onnx2tf:
        float_tflite = step3_onnx2tf(a)
    else:
        cands = glob.glob(os.path.join(a.saved_model, "*float32*.tflite"))
        float_tflite = cands[0] if cands else os.path.join(a.saved_model, "float32.tflite")
    step4_int8(a, float_tflite)
    step5_verify_int8(a)


if __name__ == "__main__":
    main()
