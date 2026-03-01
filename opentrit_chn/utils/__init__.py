# -*- coding: utf-8 -*-
"""工具模块 - 双分支共享
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
from opentrit_chn.utils.quantization_utils import calculate_entropy, apply_ternary_quantization
from opentrit_chn.utils.risk_utils import RiskLevel, detect_device_risk, analyze_deploy_risk, save_risk_report
from opentrit_chn.utils.deep_inspection import DeepInspection