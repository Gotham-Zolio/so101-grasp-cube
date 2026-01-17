#!/bin/bash
cd /home/gotham/shared/so101-grasp-cube
CUDA_VISIBLE_DEVICES=4 uv run python scripts/train_act_real_data.py --task lift --epochs 1 --batch-size 4 2>&1 | head -200
