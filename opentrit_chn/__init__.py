# -*- coding: utf-8 -*-
"""
OpenTrit-CHN v2.1.0
双分支三值神经网络量化框架：
- XC分支:适配国内信创生态(鲲鹏/昇腾/统信/麒麟)
- Global分支:适配国际主流环境(x86/NVIDIA/通用Linux)
Copyright (C) 2026 jakerdliu
Author: jakerdliu
Email: koball@263.net
"""
__version__ = "2.1.0"
__author__ = "jakerdliu"
__email__ = "koball@263.net"

# 核心模块导出
from opentrit_chn.core.trit_tensor import BaseTritTensor

# XC分支导出
from opentrit_chn.xc.trit_tensor_xc import TritTensorXC
from opentrit_chn.xc.scheduler_xc import HeterogeneousTaskSchedulerXC

# Global分支导出
from opentrit_chn.global.trit_tensor_global import TritTensorGlobal
from opentrit_chn.global.scheduler_global import HeterogeneousTaskSchedulerGlobal

# 工具类导出
from opentrit_chn.utils.quantization_utils import calculate_entropy, apply_ternary_quantization

# 新增异步/风险/巡检模块导出
from opentrit_chn.core.async_base import AsyncDeployBase
from opentrit_chn.xc.async_deploy_xc import AsyncDeployXC
from opentrit_chn.global.async_deploy_global import AsyncDeployGlobal
from opentrit_chn.utils.risk_utils import RiskLevel, detect_device_risk, analyze_deploy_risk
from opentrit_chn.utils.deep_inspection import DeepInspection