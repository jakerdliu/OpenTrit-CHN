# -*- coding: utf-8 -*-
"""
OpenTrit-CHN-Global Async Deployment Module
Async inference on NVIDIA GPU with risk control
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import asyncio
import torch
import torch.nn as nn
from opentrit_chn.core.async_base import AsyncDeployBase
from opentrit_chn.global.trit_tensor_global import TritTensorGlobal
from opentrit_chn.utils.risk_utils import RiskLevel, detect_device_risk

class AsyncDeployGlobal(AsyncDeployBase):
    """NVIDIA GPU async deployment implementation"""
    def __init__(self, model: nn.Module, device_id: int = 0, timeout: int = 30):
        super().__init__(model, device_id, timeout)
        # NVIDIA async config
        self.model = self.model.to(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
        self.model.eval()
        # Pre-detect device risk
        self.device_risk = detect_device_risk("nvidia", device_id)

    async def async_infer(self, input_data: torch.Tensor) -> torch.Tensor:
        """NVIDIA GPU async inference implementation"""
        # Risk pre-check: high risk -> raise error
        if self.device_risk["level"] >= RiskLevel.HIGH.value:
            raise RuntimeError(f"High risk on NVIDIA device {self.device_id}: {self.device_risk['reason']}")
        
        # Async inference (non-blocking)
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            self._sync_infer,
            input_data
        )
        return result

    @torch.no_grad()
    def _sync_infer(self, input_data: torch.Tensor) -> torch.Tensor:
        """Sync inference (wrapped for async)"""
        device = f"cuda:{self.device_id}" if torch.cuda.is_available() else "cpu"
        input_data = input_data.to(device)
        # Ternary tensor inference
        trit_tensor = TritTensorGlobal(input_data.cpu().numpy(), threshold=0.1)
        trit_tensor.to_nvidia(self.device_id)
        return self.model(trit_tensor.values)