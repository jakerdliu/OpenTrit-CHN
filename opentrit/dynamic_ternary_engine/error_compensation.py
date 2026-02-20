import numpy as np

def hierarchical_compensation(model, layer_name, backend="pytorch"):
    """
    分层误差补偿：GPU局部均值校准 + CPU全局误差修正
    Args:
        model: 量化后的模型
        layer_name: 目标层名称
        backend: 框架类型
    """
    # 1. GPU局部校准（均值-方差修正）
    if backend == "pytorch":
        layer = dict(model.named_parameters())[layer_name]
        quantized_data = layer.data.detach().cpu().numpy()
    elif backend == "tensorflow":
        layer = [l for l in model.layers if l.name == layer_name][0]
        quantized_data = layer.get_weights()[0]
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    # 计算校准因子：原始数据标准差/量化数据标准差（需缓存原始数据）
    # 注：实际使用需在量化前缓存原始权重
    raw_data = quantized_data  # 此处为示例，需替换为真实原始数据
    calib_factor = np.std(raw_data) / (np.std(quantized_data) + 1e-8)
    
    # 应用校准
    quantized_data = quantized_data * calib_factor
    if backend == "pytorch":
        layer.data = layer.data * calib_factor
    elif backend == "tensorflow":
        layer.set_weights([quantized_data])

    # 2. CPU全局误差修正（最小二乘法）
    # 注：全局修正需遍历全网络，此处简化实现
    global_error = np.mean(raw_data - quantized_data)
    for param in model.parameters() if backend=="pytorch" else model.weights:
        param.data = param.data + global_error  # 全局偏移修正

    print(f"Error compensation applied to layer: {layer_name}")
