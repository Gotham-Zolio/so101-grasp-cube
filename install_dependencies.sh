#!/bin/bash
# 安装真机部署所需的所有依赖

set -e  # 任何命令失败都退出

echo "=========================================="
echo "安装 DiffusionPolicy 真机部署依赖"
echo "=========================================="

# 1. 安装 env_client 库（如果未安装）
echo ""
echo "[1/3] 安装 env_client 库..."
if [ -d "packages/env-client" ]; then
    uv pip install -e packages/env-client
    echo "✅ env_client 已安装"
else
    echo "❌ 找不到 packages/env-client 目录"
    exit 1
fi

# 2. 安装 LeRobot 库
echo ""
echo "[2/3] 安装 LeRobot 库..."
pip install lerobot[compute_metrics]
echo "✅ LeRobot 已安装"

# 3. 验证安装
echo ""
echo "[3/3] 验证安装..."
python -c "import env_client; print('✅ env_client 导入成功')" || echo "❌ env_client 导入失败"
python -c "import lerobot; print('✅ lerobot 导入成功')" || echo "❌ lerobot 导入失败"

echo ""
echo "=========================================="
echo "✅ 所有依赖安装完成！"
echo "=========================================="
echo ""
echo "现在可以运行服务器："
echo "  uv run python grasp_cube/real/serve_diffusion_policy.py \\"
echo "      --policy.path checkpoints/lift_real/checkpoint-best \\"
echo "      --policy.task lift"
