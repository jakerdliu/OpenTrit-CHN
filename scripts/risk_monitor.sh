#!/bin/bash
# 风险监控脚本
# Author: jakerdliu
# Email: koball@263.net
# GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
set -e

# 配置项
PROJECT_DIR="$HOME/Desktop/OpenTrit-CHN"
DEVICE_TYPE="ascend"
DEVICE_ID=0
MONITOR_INTERVAL=10

echo "🚨 启动OpenTrit-CHN风险监控..."
echo "📌 监控间隔：$MONITOR_INTERVAL秒"

while true; do
    # 执行风险检测
    RISK_LEVEL=$(python3 - << EOF
import sys
sys.path.append("$PROJECT_DIR")
from opentrit_chn.utils.risk_utils import detect_device_risk
risk = detect_device_risk("$DEVICE_TYPE", $DEVICE_ID)
print(risk["level"])
EOF
    )

    # 风险告警
    if [ "$RISK_LEVEL" -ge 3 ]; then
        echo "⚠️  $(date) - 高风险检测到！风险等级：$RISK_LEVEL"
        # 可添加邮件/短信告警逻辑
    else
        echo "✅ $(date) - 风险等级：$RISK_LEVEL，正常运行"
    fi

    sleep $MONITOR_INTERVAL
done