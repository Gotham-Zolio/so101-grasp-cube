#!/bin/bash
# docker_entrypoint.sh
# DiffusionPolicy Server 的 Docker 启动脚本

set -e

# 从环境变量读取配置，或使用默认值
TASK=${TASK:-lift}
MODEL_PATH=${MODEL_PATH:-checkpoints/lift_real/checkpoint-best}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
DEVICE=${DEVICE:-cuda}

echo "=========================================="
echo "Starting DiffusionPolicy Policy Server"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Task: $TASK"
echo "  Model Path: $MODEL_PATH"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Device: $DEVICE"
echo ""

# 启动服务器
exec python grasp_cube/real/serve_diffusion_policy.py \
    --policy.path "$MODEL_PATH" \
    --policy.task "$TASK" \
    --host "$HOST" \
    --port "$PORT" \
    --device "$DEVICE"
