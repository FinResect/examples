import torch

model_path = "/root/gpufree-data/FPGA/examples/fast_neural_style/model/one_last_kiss_style.model"
checkpoint = torch.load(model_path, map_location="cpu")

if isinstance(checkpoint, dict):
    # 有些模型会包一层 state_dict 或 model
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    total_params = sum(v.numel() for v in state_dict.values() if torch.is_tensor(v))
else:
    total_params = sum(p.numel() for p in checkpoint.parameters())

print(f"总参数量: {total_params}")
print(f"约 {total_params / 1e6:.3f} M")
print(f"是否小于 500K: {total_params < 500_000}")
