"""
Evaluate/Deploy trained Diffusion Policy on real robot.

This script provides a unified interface for evaluating policies on real hardware
for various tasks (pick_cube, lift_cube, stack_cube, sort_cube).

It uses the Action Chunking mechanism: the policy predicts a sequence of actions,
which are then executed sequentially by the environment wrapper.

Usage Examples:

1. Pick Cube (Single Arm):
   python scripts/eval_real_policy.py \
     --policy.path checkpoints/pick_cube \
     --env.camera-config-path configs/camera_single.json \
     --env.robot so101-follower-config --env.robot.id eai-robot-01 \
     --task-name pick_cube

2. Sort Cube (Dual Arm):
   python scripts/eval_real_policy.py \
     --policy.path checkpoints/sort_cube \
     --env.camera-config-path configs/camera_dual.json \
     --env.robot bi-so101-follower-config --env.robot.id eai-robot-bi \
     --task-name sort_cube
"""

import tyro
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
import time
import torch
import numpy as np
import collections

try:
    from grasp_cube.real.lerobot_env import LeRobotEnv, LeRobotEnvConfig
    from grasp_cube.real.diffusion_policy import LeRobotDiffusionPolicy, LeRobotDiffusionPolicyConfig
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot or grasp_cube.real modules not found.")

@dataclass
class DeployConfig:
    # Environment configuration (Robot, Camera, Teleop)
    env: LeRobotEnvConfig
    
    # Policy configuration (Model path, device)
    policy: LeRobotDiffusionPolicyConfig
    
    # Evaluation settings
    num_episodes: int = 10
    max_steps: int = 600
    
    # Task name override (if not specified in env config)
    # Tasks: pick_cube, lift_cube, stack_cube, sort_cube
    task_name: Optional[str] = None
    
    # Whether to just print actions without executing (for debugging)
    dry_run: bool = False

def main(cfg: DeployConfig):
    if not LEROBOT_AVAILABLE:
        print("Error: LeRobot or grasp_cube.real dependencies not available.")
        print("Please ensure you are in the correct environment and packages are installed.")
        return

    # Update task if provided (overrides env config)
    if cfg.task_name:
        cfg.env.task = cfg.task_name
    
    print(f"Initializing for task: {cfg.env.task}")
    print("=" * 50)
    print(f"Robot Config: {type(cfg.env.robot).__name__}")
    print(f"Policy Path: {cfg.policy.path}")
    print("=" * 50)

    # Initialize environment
    # This will connect to the real robot
    try:
        env = LeRobotEnv(cfg.env)
        print("Environment initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize environment: {e}")
        return
    
    # Initialize Policy
    try:
        policy = LeRobotDiffusionPolicy(cfg.policy)
        print("Policy loaded successfully.")
    except Exception as e:
        print(f"Failed to load policy: {e}")
        env.close()
        return
    
    print("\nStarting evaluation loop...")
    print("Press Ctrl+C to stop.")
    
    try:
        for episode in range(cfg.num_episodes):
            print(f"\n--- Episode {episode + 1}/{cfg.num_episodes} ---")
            
            # Reset environment
            print("Resetting environment...")
            obs, info = env.reset()
            
            # Reset action plan queue
            action_plan = collections.deque()
            
            done = False
            step = 0
            
            start_time = time.time()
            
            while not done and step < cfg.max_steps:
                # If plan is empty, query policy for new chunk
                if not action_plan:
                    # Policy inference
                    # Returns a chunk of actions (List or Array)
                    action_chunk = policy.get_actions(obs)
                    action_plan.extend(action_chunk)
                    
                    if cfg.dry_run and step % 30 == 0:
                        print(f"Plan extended. Current plan size: {len(action_plan)}")

                # Get next action from plan
                action = action_plan.popleft()
                
                # Execute action
                obs, reward, done, truncated, info = env.step(action)
                
                # LeRobotEnv.step sets 'done' based on time limit, we can also check max_steps
                step += 1
                
                if cfg.dry_run and step % 30 == 0:
                    print(f"Step {step}: {info}")
            
            duration = time.time() - start_time
            print(f"Episode finished. Steps: {step}, Duration: {duration:.2f}s")
            
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Closing environment...")
        env.close()
        print("Done.")

if __name__ == "__main__":
    # Use tyro for nice CLI generation
    cfg = tyro.cli(DeployConfig)
    main(cfg)
