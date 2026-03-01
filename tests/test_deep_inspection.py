# -*- coding: utf-8 -*-
"""深度巡检测试
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import unittest
import numpy as np
from opentrit_chn.xc.async_deploy_xc import AsyncDeployXC
from opentrit_chn.utils.deep_inspection import DeepInspection
import mindspore as ms

class TestDeepInspection(unittest.TestCase):
    def test_inspection_precision(self):
        """测试精度巡检"""
        # 初始化部署实例
        model = ms.nn.Conv2d(3, 64, 7)
        async_deploy = AsyncDeployXC(model, device_id=0)
        inspector = DeepInspection(async_deploy, "ascend", 0)

        # 测试无数据时的精度巡检
        precision = inspector.inspect_precision()
        self.assertEqual(precision["precision"], None)

        # 测试有数据时的精度巡检
        ref_data = np.array([-1, 0, 1, 0, 1])
        infer_data = np.array([-1, 0, 1, 0, 0])
        inspector.update_precision_data(ref_data, infer_data)
        precision = inspector.inspect_precision()
        self.assertEqual(precision["precision"], 0.8)

if __name__ == "__main__":
    unittest.main()