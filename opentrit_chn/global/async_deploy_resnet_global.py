# -*- coding: utf-8 -*-
"""
Global Branch Async Deployment Example: ResNet50 + NVIDIA GPU + Deep Inspection
Copyright (C) 2026 jakerdliu (Email: koball@263.net)
GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
"""
import asyncio
import torch
import numpy as np
from opentrit_chn.global.async_deploy_global import AsyncDeployGlobal
from opentrit_chn.utils.deep_inspection import DeepInspection

# Load quantized ResNet50 model (PyTorch)
def load_quantized_resnet50():
    """Load quantized ResNet50 model"""
    # Simulate model (replace with real quantized model)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 7, padding=3),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(3, stride=2, padding=1)
    )
    # Bind quant status method (for inspection)
    model.get_quant_status = lambda: {"quantized": True, "threshold": 0.1}
    model.get_device_info = lambda: {"device": "cuda:0", "shape": (1, 64, 112, 112)}
    return model

# Async inference callback
def infer_callback(result: torch.Tensor):
    """Async inference result callback"""
    print(f"✅ Async inference completed, result shape: {result.shape}")

async def main():
    # 1. Initialize model and async deploy instance
    model = load_quantized_resnet50()
    async_deploy = AsyncDeployGlobal(model, device_id=0, timeout=30)

    # 2. Create test input
    input_data = torch.randn(1, 3, 224, 224)

    # 3. Start async inference task
    print("🚀 Starting async inference on NVIDIA GPU...")
    task = asyncio.create_task(async_deploy.run_async_task(input_data, infer_callback))

    # 4. Start deep inspection (background monitoring)
    inspector = DeepInspection(
        deploy_instance=async_deploy,
        device_type="nvidia",
        device_id=0,
        inspection_interval=2  # Inspect every 2 seconds
    )
    # Simulate reference output (for precision inspection)
    with torch.no_grad():
        reference_output = model(input_data).cpu().numpy()
    # Start inspection (non-blocking)
    import threading
    inspection_thread = threading.Thread(
        target=inspector.start_continuous_inspection
    )
    inspection_thread.daemon = True
    inspection_thread.start()

    # 5. Wait for async task completion
    try:
        result = await task
        # Update precision data to inspector
        inspector.update_precision_data(reference_output, result.cpu().numpy())
        print(f"📊 Inference result: {result[:1, :1, :1, :5]}")
    except Exception as e:
        print(f"❌ Async inference failed: {e}")
    finally:
        # Stop inspection
        inspector.stop_continuous_inspection()
        # Save inspection report
        inspector.save_inspection_history("global_inspection_report.json")
        print("📝 Inspection report saved to global_inspection_report.json")

if __name__ == "__main__":
    # Run async main program
    asyncio.run(main())