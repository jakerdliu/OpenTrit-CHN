"""
基础量化示例：将ResNet50转换为混合三值模型
"""
import torch
import opentrit

# 1. 加载原始模型
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
model.eval()

# 2. 一键量化为混合三值模型
htnn_model = opentrit.quantize(
    model,
    mode="auto",
    h1=1.2,
    h2=2.8,
    sparse_rate=0.8,
    error_compensation=True,
    backend="pytorch"
)

# 3. 可视化量化结果
weight_tensor = next(htnn_model.parameters())
opentrit.plot_tensor_distribution(weight_tensor, save_path="tensor_dist.png")

# 4. 分析量化误差
error_stats = opentrit.quantization_error_analysis(htnn_model, model, save_path="error_analysis.png")
print(f"Mean quantization error: {error_stats['mean_error']:.4f}")

# 5. 导出为ONNX（跨框架部署）
dummy_input = torch.randn(1, 3, 224, 224)
opentrit.export(htnn_model, dummy_input, "htnn_resnet50.onnx")

# 6. 异构推理
scheduler = opentrit.HeterogeneousScheduler(devices=["gpu:0", "cpu"])
output = scheduler.run(htnn_model, dummy_input)
print(f"Inference output shape: {output.shape}")
