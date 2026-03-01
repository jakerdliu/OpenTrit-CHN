# -*- coding: utf-8 -*-
"""
OpenTrit-CHN-Global Heterogeneous Task Scheduler
Task allocation for NVIDIA GPU/x86 CPU
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import numpy as np
from typing import Dict, str

class HeterogeneousTaskSchedulerGlobal:
    """Global branch heterogeneous task scheduler: NVIDIA GPU/x86 CPU"""
    def __init__(self):
        self.device_resources = {
            "nvidia": {"load": 0.0, "capacity": 100.0},
            "x86": {"load": 0.0, "capacity": 80.0}
        }

    def allocate_task(self, task_type: str, entropy: float) -> str:
        """
        Task allocation logic:
        - High entropy (important) → NVIDIA GPU
        - Low entropy → x86 CPU
        """
        if entropy > 1.0 and self.device_resources["nvidia"]["load"] < self.device_resources["nvidia"]["capacity"]:
            self.device_resources["nvidia"]["load"] += 10.0
            return "nvidia"
        else:
            self.device_resources["x86"]["load"] += 5.0
            return "x86"

    def get_device_load(self) -> Dict[str, float]:
        """Get device load (for inspection)"""
        return {k: v["load"]/v["capacity"] for k, v in self.device_resources.items()}