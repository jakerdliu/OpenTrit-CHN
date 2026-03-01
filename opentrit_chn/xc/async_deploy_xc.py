# -*- coding: utf-8 -*-
"""
OpenTrit-CHN-XC 异步部署模块
适配昇腾NPU异步推理,含风险控制
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import asyncio
import mindspore as ms
import mindspore.ops as ops
from opentrit_chn.core.async_base import AsyncDeployBase
from opentrit_chn.xc.trit_tensor_xc import TritTensorXC
from opentrit_chn.utils.risk_utils import RiskLevel, detect_device_risk

class AsyncDeployXC(AsyncDeployBase):
    """昇腾NPU异步部署实现"""
    def __init__(self, model, device_id: int = 0, timeout: int = 30):
        super().__init__(model, device_id, timeout)
        # 昇腾异步配置
        self.model = self.model.to(f"ascend:{device_id}")
        self.async_op = ops.AsyncOp()  # MindSpore异步算子
        # 预检测设备风险
        self.device_risk = detect_device_risk("ascend", device_id)

    async def async_infer(self, input_data: ms.Tensor) -> ms.Tensor:
        """昇腾NPU异步推理实现"""
        # 风险前置检查：设备不可用则直接抛出风险
        if self.device_risk["level"] >= RiskLevel.HIGH.value:
            raise RuntimeError(f"High risk on Ascend device {self.device_id}: {self.device_risk['reason']}")
        
        # 异步推理（非阻塞执行）
        loop = asyncio.get_running_loop()
        # 把MindSpore推理任务放到异步线程执行
        result = await loop.run_in_executor(
            None,
            self._sync_infer,  # 同步推理函数封装为异步
            input_data
        )
        return result

    def _sync_infer(self, input_data: ms.Tensor) -> ms.Tensor:
        """同步推理函数（供异步封装）"""
        input_data = input_data.to(f"ascend:{self.device_id}")
        # 三值张量推理
        trit_tensor = TritTensorXC(input_data.asnumpy(), threshold=0.1)
        trit_tensor.to_ascend(self.device_id)
        return self.model(trit_tensor.values)