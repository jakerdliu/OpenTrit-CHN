# -*- coding: utf-8 -*-
"""
OpenTrit-CHN-XC 异构任务调度器
适配昇腾NPU/鲲鹏CPU异构调度
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import numpy as np
from typing import Dict, str

class HeterogeneousTaskSchedulerXC:
    """信创分支异构任务调度器:分配任务到昇腾NPU/鲲鹏CPU"""
    def __init__(self):
        self.device_resources = {
            "ascend": {"load": 0.0, "capacity": 100.0},
            "kunpeng": {"load": 0.0, "capacity": 80.0}
        }

    def allocate_task(self, task_type: str, entropy: float) -> str:
        """
        任务分配逻辑：
        - 高熵值（重要）任务 → 昇腾NPU
        - 低熵值任务 → 鲲鹏CPU
        """
        # 熵值越高，任务越重要
        if entropy > 1.0 and self.device_resources["ascend"]["load"] < self.device_resources["ascend"]["capacity"]:
            self.device_resources["ascend"]["load"] += 10.0
            return "ascend"
        else:
            self.device_resources["kunpeng"]["load"] += 5.0
            return "kunpeng"

    def get_device_load(self) -> Dict[str, float]:
        """获取设备负载（供巡检调用）"""
        return {k: v["load"]/v["capacity"] for k, v in self.device_resources.items()}