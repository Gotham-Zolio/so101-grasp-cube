#!/bin/bash
# 离线测试脚本（验证推理引擎）

echo "=========================================="
echo "离线推理测试 - 检查模型是否可用"
echo "=========================================="
echo

# 测试推理引擎
echo "[1/3] 测试推理引擎..."
uv run python scripts/test_offline_inference.py
if [ $? -ne 0 ]; then
    echo "❌ 推理测试失败！"
    exit 1
fi

echo
echo "[2/3] 测试策略控制器..."
uv run python scripts/robot_policy_controller.py
if [ $? -ne 0 ]; then
    echo "❌ 控制器测试失败！"
    exit 1
fi

echo
echo "[3/3] 生成性能报告..."
uv run python -c "
import sys
sys.path.insert(0, '.')
from scripts.inference_engine import load_multi_task_models

models = load_multi_task_models()
print('\n========== 模型配置摘要 ==========')
for task, engine in models.items():
    print(f'\n{task.upper()}:')
    print(f'  State: {engine.state_dim}D')
    print(f'  Action: {engine.action_dim}D')
    print(f'  Horizon: {engine.horizon}')
"

echo
echo "=========================================="
echo "✓ 所有离线测试通过！"
echo "=========================================="
echo "接下来可以：1. 检查真机硬件"
echo "          2. 运行安全测试（机器人断电）"
echo "          3. 进行真机测试"
