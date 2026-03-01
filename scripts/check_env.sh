#!/bin/bash
# 环境检查脚本
# Author: jakerdliu
# Email: koball@263.net
# GitHub: https://github.com/jakerdliu/OpenTrit-CHN.git
set -e

echo "===== OpenTrit-CHN 环境检查 ====="

# 检查Python版本
PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "Python版本: $PYTHON_VERSION"
if [[ $(echo "$PYTHON_VERSION >= 3.10" | bc) -ne 1 ]]; then
    echo "错误：需要Python 3.10+，当前版本为$PYTHON_VERSION"
    exit 1
fi

# 检查依赖
echo -e "\n检查基础依赖..."
REQUIREMENTS=("numpy" "torch" "psutil")
for req in "${REQUIREMENTS[@]}"; do
    if python3 -c "import $req" 2>/dev/null; then
        VERSION=$(python3 -c "import $req; print($req.__version__)")
        echo "✅ $req: $VERSION"
    else
        echo "❌ $req 未安装"
    fi
done

echo -e "\n===== 环境检查完成 ====="