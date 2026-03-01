# -*- coding: utf-8 -*-
"""
三值张量（NVIDIA 版本）
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
"""
import numpy as np
import torch


class TritTensorGlobal:
    """
    三值张量实现（NVIDIA GPU 版本）
    将输入张量量化为 -1, 0, 1 三个值
    """
    def __init__(self, data: np.ndarray, threshold: float = 0.1):
        self.data = data
        self.threshold = threshold
        self.values = self._quantize()
    
    def _quantize(self) -> np.ndarray:
        """
        执行三值量化
        """
        # 三值量化: < -threshold → -1, > threshold → 1, 否则 0
        quantized = np.zeros_like(self.data)
        quantized[self.data < -self.threshold] = -1
        quantized[self.data > self.threshold] = 1
        return quantized
    
    def to_nvidia(self, device_id: int) -> None:
        """
        将张量转移到 NVIDIA GPU
        """
        if torch.cuda.is_available():
            self.values = torch.tensor(self.values, dtype=torch.float32).to(f"cuda:{device_id}")
        else:
            self.values = torch.tensor(self.values, dtype=torch.float32)
    
    def cpu(self) -> np.ndarray:
        """
        将张量转移到 CPU
        """
        if isinstance(self.values, torch.Tensor):
            return self.values.cpu().numpy()
        return self.values
