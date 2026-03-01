# -*- coding: utf-8 -*-
"""
OpenTrit-CHN-XC 风险分析模块
聚焦昇腾NPU/鲲鹏CPU部署风险
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
from typing import Dict, Any
from opentrit_chn.utils.risk_utils import RiskLevel, detect_device_risk, analyze_deploy_risk

class RiskAnalysisXC:
    """信创分支风险分析类"""
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.risk_history = []

    def full_risk_analysis(self, deploy_status: Dict[str, Any]) -> Dict[str, Any]:
        """完整风险分析：设备+部署+性能"""
        # 1. 设备风险
        device_risk = detect_device_risk("ascend", self.device_id)
        # 2. 部署风险
        deploy_risk = analyze_deploy_risk(deploy_status)
        # 3. 综合风险
        overall_risk = max(device_risk["level"], deploy_risk["risk_level"])
        
        risk_result = {
            "timestamp": deploy_status.get("end_time", deploy_status.get("start_time")),
            "device_risk": device_risk,
            "deploy_risk": deploy_risk,
            "overall_risk": overall_risk,
            "suggestion": self._get_suggestion(overall_risk)
        }
        self.risk_history.append(risk_result)
        return risk_result

    def _get_suggestion(self, risk_level: int) -> str:
        """根据风险等级返回建议"""
        if risk_level == RiskLevel.LOW.value:
            return "正常运行，无需处理"
        elif risk_level == RiskLevel.MEDIUM.value:
            return "建议降低批量大小或优化模型量化阈值"
        elif risk_level == RiskLevel.HIGH.value:
            return "立即检查昇腾设备状态，重启驱动或降低负载"
        else:
            return "紧急：昇腾设备不可用，切换至鲲鹏CPU或修复硬件"

    def get_risk_history(self) -> list:
        """获取风险分析历史"""
        return self.risk_history