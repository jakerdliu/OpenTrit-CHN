#!/bin/bash
# 启动OpenTrit-CHN深度巡检脚本
# Author: jakerdliu
# Email: koball@263.net
# GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
set -e

# 配置项
PROJECT_DIR="$HOME/Desktop/OpenTrit-CHN"
DEVICE_TYPE="ascend"  # 可选：ascend/nvidia
DEVICE_ID=0
INSPECTION_INTERVAL=5  # 巡检间隔（秒）

# 进入项目目录
cd $PROJECT_DIR

# 检查Python环境
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc) -ne 1 ]]; then
    echo "错误：需要Python 3.10+，当前版本为$PYTHON_VERSION"
    exit 1
fi

# 安装依赖（若未安装）
pip install -r requirements.txt > /dev/null 2>&1

# 启动巡检（调用Python脚本）
echo "🚀 启动OpenTrit-CHN深度巡检..."
echo "📌 项目目录：$PROJECT_DIR"
echo "📌 设备类型：$DEVICE_TYPE (ID: $DEVICE_ID)"
echo "📌 巡检间隔：$INSPECTION_INTERVAL秒"
echo "🔍 巡检日志将保存到 inspection_history.json"

python3 - << EOF
import sys
sys.path.append("$PROJECT_DIR")
from opentrit_chn.utils.deep_inspection import DeepInspection
from opentrit_chn.xc.async_deploy_xc import AsyncDeployXC
import mindspore as ms

# 初始化空的部署实例（仅用于巡检）
model = ms.nn.Conv2d(3, 64, 7)
async_deploy = AsyncDeployXC(model, device_id=$DEVICE_ID)

# 启动巡检
inspector = DeepInspection(
    deploy_instance=async_deploy,
    device_type="$DEVICE_TYPE",
    device_id=$DEVICE_ID,
    inspection_interval=$INSPECTION_INTERVAL
)
inspector.start_continuous_inspection()
EOF

echo "🛑 按Ctrl+C停止巡检"
trap "echo '⏹ 巡检已停止'; exit 0" SIGINT
while true; do sleep 1; done