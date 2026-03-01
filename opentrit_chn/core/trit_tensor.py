# -*- coding: utf-8 -*-
"""
OpenTrit-CHN Core: Hybrid Ternary Tensor Implementation
Shared core module for XC/Global branches
Copyright (C) 2026 jakerdliu
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import numpy as np
from typing import Dict, Any, Optional

class BaseTritTensor:
    """Base class for ternary tensor (shared by all branches)"""
    def __init__(self, values, threshold, mode="symmetric", error_compensation_params=None):
        self.values = self._validate_values(values)
        self.threshold = threshold
        self.mode = mode
        self.error_compensation_params = error_compensation_params or {"correction_factor": 1.0}
        # 量化状态（供异步/巡检追踪）
        self.quant_status: Dict[str, Any] = {
            "quantized": True,
            "threshold_used": threshold,
            "mode_used": mode,
            "error_compensation_applied": False
        }

    def _validate_values(self, values):
        """Validate tensor values are in {-1, 0, 1}"""
        values = np.array(values)
        if not np.all(np.isin(values, [-1, 0, 1])):
            raise ValueError("Ternary tensor values must be -1, 0, or 1")
        return values

    def apply_error_compensation(self, data):
        """Apply hierarchical error compensation (shared logic)"""
        compensated_data = data * self.error_compensation_params["correction_factor"]
        self.quant_status["error_compensation_applied"] = True
        return compensated_data
    
    def get_quant_status(self) -> Dict[str, Any]:
        """获取量化状态（供巡检调用）"""
        return self.quant_status