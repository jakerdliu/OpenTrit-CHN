# -*- coding: utf-8 -*-
"""通用量化工具类 - 双分支共享
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import numpy as np

def calculate_entropy(weight_tensor):
    """计算权重张量的信息熵"""
    unique_vals, counts = np.unique(weight_tensor, return_counts=True)
    total = counts.sum()
    probs = counts / total
    entropy = -np.sum(probs * np.log2(probs))
    return entropy

def apply_ternary_quantization(tensor, threshold, mode="symmetric"):
    """通用三值量化函数"""
    if mode == "symmetric":
        ternary = np.where(tensor > threshold, 1, np.where(tensor < -threshold, -1, 0))
    else:  # asymmetric
        ternary = np.where(tensor > threshold, 1, np.where(tensor < 0, -1, 0))
    return ternary