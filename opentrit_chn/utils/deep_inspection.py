# -*- coding: utf-8 -*-
"""
深度巡检工具 - 双分支共享
监控部署状态、性能、精度、风险
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import time
import json
import psutil
import numpy as np
from typing import Dict, List, Any, Optional
from opentrit_chn.utils.risk_utils import detect_device_risk, analyze_deploy_risk, RiskLevel
from opentrit_chn.core.async_base import AsyncDeployBase

class DeepInspection:
    """深度巡检类：全维度监控部署状态"""
    def __init__(self, 
                 deploy_instance: AsyncDeployBase,
                 device_type: str,
                 device_id: int = 0,
                 inspection_interval: int = 5,  # 巡检间隔（秒）
                 config_path: str = "configs/inspection_config.json"):
        self.deploy_instance = deploy_instance  # 异步部署实例
        self.device_type = device_type
        self.device_id = device_id
        self.inspection_interval = inspection_interval
        self.config = self.load_config(config_path)
        self.inspection_history: List[Dict[str, Any]] = []  # 巡检历史（供分析）
        self.running = False
        # 精度缓存（解决异步结果延迟问题）
        self.reference_output_cache = None
        self.infer_output_cache = None

    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载巡检配置：阈值、监控项"""
        default_config = {
            "monitor_items": ["device_status", "task_status", "performance", "precision", "risk"],
            "thresholds": {
                "latency_warn": 10.0,  # 延迟警告阈值（s）
                "cpu_warn": 90.0,       # CPU使用率警告阈值（%）
                "mem_warn": 90.0,       # 内存使用率警告阈值（%）
                "precision_warn": 0.85  # 精度警告阈值
            }
        }
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            # 保存默认配置
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(default_config, f, ensure_ascii=False, indent=4)
            return default_config

    def inspect_device_status(self) -> Dict[str, Any]:
        """巡检1：设备状态（硬件+驱动）"""
        device_risk = detect_device_risk(self.device_type, self.device_id)
        # 系统资源
        sys_info = {
            "cpu_usage": psutil.cpu_percent(),
            "mem_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent
        }
        # 获取张量设备信息（补充巡检维度）
        tensor_device_info = None
        if hasattr(self.deploy_instance.model, "get_device_info"):
            tensor_device_info = self.deploy_instance.model.get_device_info()
        
        return {
            "device_type": self.device_type,
            "device_id": self.device_id,
            "device_risk": device_risk,
            "system_resources": sys_info,
            "tensor_device_info": tensor_device_info,
            "status": "normal" if device_risk["level"] < RiskLevel.HIGH.value else "abnormal"
        }

    def inspect_task_status(self) -> Dict[str, Any]:
        """巡检2：异步任务状态"""
        task_status = self.deploy_instance.get_task_status()
        risk_analysis = analyze_deploy_risk(task_status)
        # 获取量化状态（补充巡检维度）
        quant_status = None
        if hasattr(self.deploy_instance.model, "get_quant_status"):
            quant_status = self.deploy_instance.model.get_quant_status()
        
        return {
            "task_running": task_status["running"],
            "task_latency": task_status["latency"],
            "task_success": task_status["success"],
            "task_error": task_status["error"],
            "task_risk": risk_analysis,
            "quant_status": quant_status
        }

    def inspect_performance(self) -> Dict[str, Any]:
        """巡检3：性能指标（延迟、吞吐量）"""
        task_status = self.deploy_instance.get_task_status()
        # 吞吐量：假设已执行任务数（模拟）
        throughput = 0.0
        if task_status["latency"] > 0:
            throughput = 1 / task_status["latency"]  # QPS
        
        # 性能风险判断
        latency_warn = self.config["thresholds"]["latency_warn"]
        performance_risk = RiskLevel.LOW.value
        if task_status["latency"] > latency_warn:
            performance_risk = RiskLevel.MEDIUM.value

        return {
            "latency": task_status["latency"],
            "throughput": throughput,
            "performance_risk": performance_risk,
            "latency_warn_threshold": latency_warn
        }

    def inspect_precision(self) -> Dict[str, Any]:
        """巡检4：精度指标（修复缓存逻辑）"""
        if self.reference_output_cache is None or self.infer_output_cache is None:
            return {
                "precision": None,
                "precision_risk": RiskLevel.LOW.value,
                "message": "No reference/infer data for precision check"
            }
        
        # 计算精度（准确率）
        reference_binary = np.where(self.reference_output_cache != 0, 1, 0)
        infer_binary = np.where(self.infer_output_cache != 0, 1, 0)
        accuracy = np.sum(reference_binary == infer_binary) / reference_binary.size
        
        # 精度风险判断
        precision_warn = self.config["thresholds"]["precision_warn"]
        precision_risk = RiskLevel.LOW.value
        if accuracy < precision_warn:
            precision_risk = RiskLevel.HIGH.value

        return {
            "precision": accuracy,
            "precision_risk": precision_risk,
            "precision_warn_threshold": precision_warn
        }

    def update_precision_data(self, reference_output: np.ndarray, infer_output: np.ndarray):
        """更新精度数据（解决异步结果延迟问题）"""
        self.reference_output_cache = reference_output
        self.infer_output_cache = infer_output

    def run_inspection(self) -> Dict[str, Any]:
        """执行一次完整巡检（修复参数传递）"""
        inspection_result = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "device_status": self.inspect_device_status() if "device_status" in self.config["monitor_items"] else None,
            "task_status": self.inspect_task_status() if "task_status" in self.config["monitor_items"] else None,
            "performance": self.inspect_performance() if "performance" in self.config["monitor_items"] else None,
            "precision": self.inspect_precision() if "precision" in self.config["monitor_items"] else None,
            "overall_risk": RiskLevel.LOW.value
        }

        # 计算整体风险（取最高风险等级）
        risk_levels = []
        if inspection_result["device_status"]:
            risk_levels.append(inspection_result["device_status"]["device_risk"]["level"])
        if inspection_result["task_status"]:
            risk_levels.append(inspection_result["task_status"]["task_risk"]["risk_level"])
        if inspection_result["performance"]:
            risk_levels.append(inspection_result["performance"]["performance_risk"])
        if inspection_result["precision"]:
            risk_levels.append(inspection_result["precision"]["precision_risk"])
        
        if risk_levels:
            inspection_result["overall_risk"] = max(risk_levels)

        # 记录巡检历史
        self.inspection_history.append(inspection_result)
        return inspection_result

    def start_continuous_inspection(self):
        """启动持续巡检（修复参数问题）"""
        self.running = True
        print(f"Starting deep inspection (interval: {self.inspection_interval}s)...")
        while self.running:
            result = self.run_inspection()
            # 打印巡检结果（或写入日志）
            print(f"Inspection [{result['timestamp']}] - Overall Risk: {result['overall_risk']}")
            if result["overall_risk"] >= RiskLevel.HIGH.value:
                print(f"⚠️  High risk detected: {json.dumps(result, ensure_ascii=False, indent=2)}")
            # 保存巡检历史
            self.save_inspection_history()
            time.sleep(self.inspection_interval)

    def stop_continuous_inspection(self):
        """停止持续巡检"""
        self.running = False
        print("Deep inspection stopped.")

    def save_inspection_history(self, path: str = "inspection_history.json"):
        """保存巡检历史到文件"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.inspection_history, f, ensure_ascii=False, indent=4)