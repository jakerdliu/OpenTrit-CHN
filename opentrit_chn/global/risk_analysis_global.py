# -*- coding: utf-8 -*-
"""
OpenTrit-CHN-Global Risk Analysis Module
Focus on NVIDIA GPU/x86 CPU deployment risks
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
from typing import Dict, Any
from opentrit_chn.utils.risk_utils import RiskLevel, detect_device_risk, analyze_deploy_risk

class RiskAnalysisGlobal:
    """Global branch risk analysis class"""
    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.risk_history = []

    def full_risk_analysis(self, deploy_status: Dict[str, Any]) -> Dict[str, Any]:
        """Full risk analysis: device + deployment + performance"""
        # 1. Device risk
        device_risk = detect_device_risk("nvidia", self.device_id)
        # 2. Deployment risk
        deploy_risk = analyze_deploy_risk(deploy_status)
        # 3. Overall risk
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
        """Get suggestion based on risk level"""
        if risk_level == RiskLevel.LOW.value:
            return "Normal operation, no action needed"
        elif risk_level == RiskLevel.MEDIUM.value:
            return "Suggest reduce batch size or optimize quantization threshold"
        elif risk_level == RiskLevel.HIGH.value:
            return "Check NVIDIA device status immediately, restart driver or reduce load"
        else:
            return "Emergency: NVIDIA device unavailable, switch to x86 CPU or fix hardware"

    def get_risk_history(self) -> list:
        """Get risk analysis history"""
        return self.risk_history