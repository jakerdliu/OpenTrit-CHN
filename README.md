# OpenTrit-CHN v2.0.0

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![Release](https://img.shields.io/badge/Release-2026.Q2-orange.svg)](configs/version.json)

OpenTrit-CHN 是基于 OpenTrit 升级的双分支三值神经网络量化框架，同时适配**国内信创生态**和**国际主流环境**，新增**异步部署**、**风险分析**、**深度巡检**核心能力。

## 核心特性
- **双分支设计**：
  - XC分支：适配鲲鹏/昇腾/统信/麒麟等国产软硬件，中文注释，符合GB/T 35274-2023标准；
  - Global分支：适配x86/NVIDIA/通用Linux，英文注释，符合IEEE 1855-2019标准。
- **异步部署**：非阻塞推理，支持超时控制、任务状态追踪，大幅提升部署吞吐量；
- **风险分析**：设备风险、部署风险、性能风险全维度识别，提供分级预警和兜底方案；
- **深度巡检**：实时监控设备状态、任务性能、精度指标，生成可视化巡检报告；
- **统一核心算法**：共享三值量化、熵值模式切换、分层误差补偿等核心逻辑；
- **易用性**：极简API，一行代码完成模型量化与跨框架转换；
- **高性能**：16×内存压缩比，3.2×推理加速，82.5%硬件利用率。

## 快速开始
### 环境准备
```bash
# 克隆仓库
git clone https://github.com/jakerdliu/OpenTrit-CHN.git
cd ~/Desktop/OpenTrit-CHN

# 安装基础依赖
pip install -r requirements.txt