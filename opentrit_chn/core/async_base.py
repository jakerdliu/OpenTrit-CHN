# -*- coding: utf-8 -*-
"""
异步部署基类 - 双分支共享
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Optional

class AsyncDeployBase(ABC):
    """异步部署抽象基类：定义统一异步接口"""
    def __init__(self, model, device_id: int = 0, timeout: int = 30):
        self.model = model  # 量化后的三值模型
        self.device_id = device_id
        self.timeout = timeout  # 异步任务超时阈值（风险控制）
        self.task_status: Dict[str, Any] = {
            "running": False,
            "start_time": None,
            "end_time": None,
            "error": None,
            "latency": 0.0,
            "success": False
        }  # 任务状态（用于巡检/风险分析）

    @abstractmethod
    async def async_infer(self, input_data: Any) -> Any:
        """异步推理核心接口（子类实现）"""
        pass

    async def run_async_task(self, input_data: Any, callback: Optional[Callable] = None) -> Any:
        """封装异步任务：含超时控制、状态记录（风险分析基础）"""
        self.task_status["running"] = True
        self.task_status["start_time"] = time.time()
        self.task_status["error"] = None
        self.task_status["success"] = False

        try:
            # 超时控制（风险兜底）
            result = await asyncio.wait_for(
                self.async_infer(input_data),
                timeout=self.timeout
            )
            self.task_status["end_time"] = time.time()
            self.task_status["latency"] = self.task_status["end_time"] - self.task_status["start_time"]
            self.task_status["success"] = True
            
            # 回调函数（异步结果处理）
            if callback:
                try:
                    callback(result)
                except Exception as cb_e:
                    # 回调异常不影响主流程，但记录
                    self.task_status["error"] = f"Callback error: {cb_e}"
            return result
        except asyncio.TimeoutError:
            self.task_status["error"] = f"Async task timeout (>{self.timeout}s)"
            raise asyncio.TimeoutError(self.task_status["error"])
        except Exception as e:
            self.task_status["error"] = str(e)
            raise RuntimeError(f"Async task failed: {e}")
        finally:
            self.task_status["running"] = False

    def get_task_status(self) -> Dict[str, Any]:
        """获取任务状态（供巡检调用）"""
        return self.task_status