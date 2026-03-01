# -*- coding: utf-8 -*-
"""
信创分支异步部署示例：ResNet50 + 昇腾NPU + 深度巡检
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
"""
import asyncio
import mindspore as ms
import numpy as np
from opentrit_chn.xc.async_deploy_xc import AsyncDeployXC
from opentrit_chn.utils.deep_inspection import DeepInspection

# 模拟加载量化后的ResNet50模型（MindSpore）
def load_quantized_resnet50():
    """加载量化后的ResNet50模型"""
    # 模拟模型（实际需替换为真实量化模型）
    model = ms.nn.SequentialCell(
        ms.nn.Conv2d(3, 64, 7, pad_mode="same"),
        ms.nn.ReLU(),
        ms.nn.MaxPool2d(3, stride=2)
    )
    # 绑定量化状态方法（供巡检调用）
    model.get_quant_status = lambda: {"quantized": True, "threshold": 0.1}
    model.get_device_info = lambda: {"device": "ascend:0", "shape": (1, 64, 112, 112)}
    return model

# 异步推理回调函数
def infer_callback(result: ms.Tensor):
    """异步推理结果回调"""
    print(f"✅ Async inference completed, result shape: {result.shape}")

async def main():
    # 1. 初始化模型和异步部署实例
    model = load_quantized_resnet50()
    async_deploy = AsyncDeployXC(model, device_id=0, timeout=30)

    # 2. 构造测试输入
    input_data = ms.Tensor(np.random.randn(1, 3, 224, 224), dtype=ms.float32)

    # 3. 启动异步推理任务
    print("🚀 Starting async inference on Ascend NPU...")
    task = asyncio.create_task(async_deploy.run_async_task(input_data, infer_callback))

    # 4. 启动深度巡检（后台监控）
    inspector = DeepInspection(
        deploy_instance=async_deploy,
        device_type="ascend",
        device_id=0,
        inspection_interval=2  # 每2秒巡检一次
    )
    # 模拟参考输出（用于精度巡检）
    reference_output = model(input_data).asnumpy()
    # 启动巡检（非阻塞）
    import threading
    inspection_thread = threading.Thread(
        target=inspector.start_continuous_inspection
    )
    inspection_thread.daemon = True
    inspection_thread.start()

    # 5. 等待异步任务完成
    try:
        result = await task
        # 更新精度数据到巡检器
        inspector.update_precision_data(reference_output, result.asnumpy())
        print(f"📊 Inference result: {result[:1, :1, :1, :5]}")
    except Exception as e:
        print(f"❌ Async inference failed: {e}")
    finally:
        # 停止巡检
        inspector.stop_continuous_inspection()
        # 保存巡检报告
        inspector.save_inspection_history("xc_inspection_report.json")
        print("📝 Inspection report saved to xc_inspection_report.json")

if __name__ == "__main__":
    # 设置MindSpore异步模式
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="Ascend", device_id=0)
    # 运行异步主程序
    asyncio.run(main())