# -*- coding: utf-8 -*-
"""
三值张量（昇腾版本）
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
"""
import numpy as np
import mindspore as ms


class TritTensorXC:
    """
    三值张量实现（昇腾NPU 版本）
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
    
    def to_ascend(self, device_id: int) -> None:
        """
        将张量转移到昇腾NPU
        """
        try:
            self.values = ms.Tensor(self.values, dtype=ms.float32)
            self.values = self.values.to(f"ascend:{device_id}")
        except Exception:
            # 如果设备不可用，保持在CPU上
            self.values = ms.Tensor(self.values, dtype=ms.float32)
    
    def cpu(self) -> np.ndarray:
        """
        将张量转移到 CPU
        """
        if isinstance(self.values, ms.Tensor):
            return self.values.asnumpy()
        return self.values
