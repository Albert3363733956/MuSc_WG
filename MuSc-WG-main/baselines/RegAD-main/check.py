import torch

# 1. 检测是否有可用的 CUDA 设备
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ 当前使用 GPU 训练: {torch.cuda.get_device_name(0)}")
    print(f"   GPU 数量: {torch.cuda.device_count()}")
else:
    device = torch.device("cpu")
    print("⚠️ 当前使用 CPU 训练 (未检测到 GPU)")

# 2. 验证：创建一个张量并移动到对应设备
x = torch.tensor([1.0, 2.0, 3.0]).to(device)
print(f"\n🔍 验证张量设备: {x.device}")