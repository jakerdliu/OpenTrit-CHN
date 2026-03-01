# -*- coding: utf-8 -*-
"""Global Branch Example: Quantize ResNet50 and deploy to NVIDIA GPU
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import torch
from opentrit_chn import TritTensorGlobal, HeterogeneousTaskSchedulerGlobal, calculate_entropy, apply_ternary_quantization

# Load pre-trained ResNet50 (PyTorch)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.eval()

# Initialize scheduler
scheduler = HeterogeneousTaskSchedulerGlobal()

# Quantize layer by layer
for name, param in model.named_parameters():
    if "conv" in name or "fc" in name:
        # Calculate entropy
        entropy = calculate_entropy(param.detach().cpu().numpy())
        # Auto quantization
        ternary_weights = apply_ternary_quantization(param.detach().cpu().numpy(), threshold=0.1, mode="auto")
        # Wrap as global ternary tensor
        trit_tensor = TritTensorGlobal(ternary_weights, threshold=0.1)
        # Migrate to NVIDIA GPU
        trit_tensor.to_nvidia(device_id=0)
        # Allocate task
        device = scheduler.allocate_task("conv" if "conv" in name else "fc", entropy)
        print(f"Layer {name}: entropy={entropy:.2f}, allocated to {device}")

print("ResNet50 quantization completed (Global version)")