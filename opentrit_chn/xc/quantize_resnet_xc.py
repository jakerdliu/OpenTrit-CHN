# -*- coding: utf-8 -*-
"""信创分支示例：量化ResNet50并部署到昇腾NPU
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import mindspore as ms
from opentrit_chn import TritTensorXC, HeterogeneousTaskSchedulerXC, calculate_entropy, apply_ternary_quantization

# 模拟加载ResNet50模型（MindSpore）
model = ms.load_checkpoint("resnet50_ms.ckpt")

# 初始化调度器
scheduler = HeterogeneousTaskSchedulerXC()

# 逐层量化
for layer_name, weights in model.items():
    if "conv" in layer_name or "fc" in layer_name:
        # 计算熵值
        entropy = calculate_entropy(weights.asnumpy())
        # 自动量化
        ternary_weights = apply_ternary_quantization(weights.asnumpy(), threshold=0.1, mode="auto")
        # 封装为信创版三值张量
        trit_tensor = TritTensorXC(ternary_weights, threshold=0.1)
        # 迁移到昇腾NPU
        trit_tensor.to_ascend(device_id=0)
        # 任务分配
        device = scheduler.allocate_task("conv" if "conv" in layer_name else "fc", entropy)
        print(f"Layer {layer_name}: entropy={entropy:.2f}, allocated to {device}")

print("ResNet50量化完成（信创版）")