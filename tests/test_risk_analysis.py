# -*- coding: utf-8 -*-
"""风险分析测试
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import unittest
from opentrit_chn.utils.risk_utils import RiskLevel, detect_device_risk, analyze_deploy_risk
from opentrit_chn.xc.risk_analysis_xc import RiskAnalysisXC

class TestRiskAnalysis(unittest.TestCase):
    def test_detect_device_risk(self):
        """测试设备风险检测"""
        # 测试昇腾设备（模拟工具不存在）
        risk = detect_device_risk("ascend", 0)
        self.assertEqual(risk["level"], RiskLevel.CRITICAL.value)

        # 测试NVIDIA设备（模拟无CUDA）
        risk = detect_device_risk("nvidia", 0)
        self.assertEqual(risk["level"], RiskLevel.CRITICAL.value)

    def test_analyze_deploy_risk(self):
        """测试部署风险分析"""
        # 测试超时风险
        deploy_status = {
            "error": "Async task timeout (>30s)",
            "latency": 35.0
        }
        risk = analyze_deploy_risk(deploy_status)
        self.assertEqual(risk["risk_level"], RiskLevel.HIGH.value)

        # 测试执行失败风险
        deploy_status = {
            "error": "Model inference failed",
            "latency": 0.0
        }
        risk = analyze_deploy_risk(deploy_status)
        self.assertEqual(risk["risk_level"], RiskLevel.CRITICAL.value)

if __name__ == "__main__":
    unittest.main()