# -*- coding: utf-8 -*-
"""异步部署测试
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
"""
import asyncio
import unittest
import mindspore as ms
import torch
import numpy as np
from opentrit_chn.xc.async_deploy_xc import AsyncDeployXC
from opentrit_chn.global.async_deploy_global import AsyncDeployGlobal

class TestAsyncDeploy(unittest.TestCase):
    def test_async_deploy_xc(self):
        """测试信创分支异步部署"""
        # 模拟模型
        model = ms.nn.Conv2d(3, 64, 7)
        async_deploy = AsyncDeployXC(model, device_id=0, timeout=5)
        input_data = ms.Tensor(np.random.randn(1, 3, 224, 224), dtype=ms.float32)

        # 运行异步任务
        async def run_test():
            try:
                result = await async_deploy.run_async_task(input_data)
                self.assertEqual(result.shape, (1, 64, 218, 218))
            except Exception as e:
                # 无昇腾设备时跳过
                self.assertTrue("Ascend tool" in str(e) or "unavailable" in str(e))

        asyncio.run(run_test())

    def test_async_deploy_global(self):
        """测试国际分支异步部署"""
        # 模拟模型
        model = torch.nn.Conv2d(3, 64, 7)
        async_deploy = AsyncDeployGlobal(model, device_id=0, timeout=5)
        input_data = torch.randn(1, 3, 224, 224)

        # 运行异步任务
        async def run_test():
            try:
                result = await async_deploy.run_async_task(input_data)
                self.assertEqual(result.shape, (1, 64, 218, 218))
            except Exception as e:
                # 无NVIDIA设备时跳过
                self.assertTrue("CUDA" in str(e) or "unavailable" in str(e))

        asyncio.run(run_test())

if __name__ == "__main__":
    unittest.main()