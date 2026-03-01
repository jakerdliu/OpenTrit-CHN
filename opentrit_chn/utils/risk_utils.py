# -*- coding: utf-8 -*-
"""
风险控制工具模块
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
"""
from enum import Enum
from typing import Dict, Any


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2


def detect_device_risk(device_type: str, device_id: int) -> Dict[str, Any]:
    """
    检测设备风险
    
    Args:
        device_type: 设备类型 ("nvidia" 或 "ascend")
        device_id: 设备ID
    
    Returns:
        包含风险等级和原因的字典
    """
    try:
        if device_type == "nvidia":
            import torch
            if not torch.cuda.is_available():
                return {
                    "level": RiskLevel.HIGH.value,
                    "reason": "NVIDIA CUDA is not available"
                }
            if device_id >= torch.cuda.device_count():
                return {
                    "level": RiskLevel.HIGH.value,
                    "reason": f"NVIDIA device {device_id} is out of range"
                }
        elif device_type == "ascend":
            try:
                import mindspore as ms
                if not ms.context.get_context("device_target") == "Ascend":
                    return {
                        "level": RiskLevel.HIGH.value,
                        "reason": "Ascend device is not available"
                    }
            except Exception:
                return {
                    "level": RiskLevel.HIGH.value,
                    "reason": "MindSpore or Ascend tool is not installed"
                }
        
        # 设备可用，返回低风险
        return {
            "level": RiskLevel.LOW.value,
            "reason": "Device is available"
        }
    except Exception as e:
        return {
            "level": RiskLevel.MEDIUM.value,
            "reason": f"Error checking device: {str(e)}"
        }
