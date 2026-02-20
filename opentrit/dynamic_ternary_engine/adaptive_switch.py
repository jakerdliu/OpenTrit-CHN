import numpy as np
from opentrit.framework_abstraction.trit_tensor import TritTensor

def calculate_entropy(data):
    """计算数据分布的信息熵，用于自动切换模式"""
    hist, _ = np.histogram(data, bins=100)
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]  # 避免log(0)
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def quantize(model, mode="auto", h1=1.2, h2=2.8, sparse_rate=0.8, 
             error_compensation=True, backend="pytorch"):
    """
    核心量化函数：一键将模型转换为混合三值神经网络
    Args:
        model: 原始模型（PyTorch/TensorFlow）
        mode: "auto"/"symmetric"/"asymmetric"/自定义层配置字典
        h1: 低熵阈值（≤h1用对称模式）
        h2: 高熵阈值（≥h2用非对称模式）
        sparse_rate: 非对称模式稀疏率
        error_compensation: 是否启用分层误差补偿
        backend: 目标框架
    Returns:
        量化后的混合三值模型
    """
    # 适配不同框架的模型遍历逻辑
    if backend == "pytorch":
        layers = list(model.named_parameters())
    elif backend == "tensorflow":
        layers = list(model.layers)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    quantized_model = model  # 基础复制，实际需根据框架深拷贝
    layer_configs = {}

    # 1. 解析量化配置
    if isinstance(mode, dict):
        # 手动层配置：{"conv1": {"mode": "symmetric"}, ...}
        layer_configs = mode
    else:
        # 自动模式：基于信息熵切换
        for name, param in layers:
            if "weight" in name or "kernel" in name:  # 仅量化权重/核
                data = param.detach().cpu().numpy() if backend=="pytorch" else param.numpy()
                entropy = calculate_entropy(data)
                if entropy <= h1:
                    layer_configs[name] = {"mode": "symmetric", "sparse_rate": 0}
                elif entropy >= h2:
                    layer_configs[name] = {"mode": "asymmetric", "sparse_rate": sparse_rate}
                else:
                    # 混合模式：线性插值稀疏率
                    layer_configs[name] = {
                        "mode": "asymmetric", 
                        "sparse_rate": sparse_rate * (entropy - h1)/(h2 - h1)
                    }

    # 2. 逐层量化
    for name, param in layers:
        if name in layer_configs:
            cfg = layer_configs[name]
            # 转换为三值Tensor
            trit_tensor = TritTensor(
                param, mode=cfg["mode"], sparse_rate=cfg["sparse_rate"], backend=backend
            )
            # 替换模型权重（需适配框架API）
            if backend == "pytorch":
                param.data = trit_tensor.to_backend()
            elif backend == "tensorflow":
                param.set_weights([trit_tensor.to_backend()])

            # 3. 启用误差补偿（可选）
            if error_compensation:
                from opentrit.dynamic_ternary_engine.error_compensation import hierarchical_compensation
                hierarchical_compensation(quantized_model, name, backend)

    print(f"Quantization completed! Layer configs: {layer_configs}")
    return quantized_model
