"""
Data collection script using motion planning solutions.
Collects trajectories for Diffusion Policy training.

This script:
1. Uses motion planning solutions to generate successful trajectories
2. Records trajectories in ManiSkill format (using RecordEpisode wrapper)
3. Trajectories can then be converted to LeRobot Dataset format using convert_trajectory_to_lerobot.py

Key features:
- Records joint positions (not end-effector poses) as actions
- Can apply camera distortion to front camera images (during conversion)
- Supports domain randomization (via environment randomization)
"""

import argparse
import pathlib
import time
from typing import Dict, Any, Optional
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import os.path as osp

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from grasp_cube.motionplanning.so101.solutions import solveLiftCube, solveStackCube, solveSortCube


# Task prompts
TASK_PROMPTS = {
    "LiftCubeSO101-v1": "Pick up the red cube and lift it.",
    "StackCubeSO101-v1": "Stack the red cube on top of the green cube.",
    "SortCubeSO101-v1": "Move the red cube to the left region and the green cube to the right region.",
}

# Motion planning solutions
MP_SOLUTIONS = {
    "LiftCubeSO101-v1": solveLiftCube,
    "StackCubeSO101-v1": solveStackCube,
    "SortCubeSO101-v1": solveSortCube,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Collect motion planning data for Diffusion Policy training")
    parser.add_argument(
        "-e", "--env-id",
        type=str,
        default="LiftCubeSO101-v1",
        choices=list(MP_SOLUTIONS.keys()),
        help="Environment ID to collect data for"
    )
    parser.add_argument(
        "-n", "--num-episodes",
        type=int,
        default=500,
        help="Number of successful episodes to collect"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets",
        help="Output directory for datasets"
    )
    parser.add_argument(
        "--apply-distortion",
        action="store_true",
        default=True,
        help="Apply camera distortion to front camera images (recommended for sim-to-real)"
    )
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        default=True,
        help="Enable domain randomization (lighting, textures, etc.)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--agent-idx",
        type=int,
        default=None,
        help="Agent index for motion planning (0=left, 1=right, None=default)"
    )
    parser.add_argument(
        "--shader",
        type=str,
        default="default",
        help="Shader pack for rendering (default, rt, rt-fast)"
    )
    return parser.parse_args()


def get_robot_state(env, agent_idx: Optional[int] = None) -> np.ndarray:
    """
    Get robot joint positions (qpos) as state.
    For dual-arm: returns concatenated [left_arm_qpos, right_arm_qpos] (12-dim)
    For single-arm: returns arm_qpos (6-dim)
    """
    env_unwrapped = env.unwrapped
    if hasattr(env_unwrapped.agent, "agents"):
        # Dual-arm setup
        left_qpos = env_unwrapped.agent.agents[0].robot.get_qpos()  # (b, 7)
        right_qpos = env_unwrapped.agent.agents[1].robot.get_qpos()  # (b, 7)
        # Extract arm joints only (first 6, excluding gripper)
        left_arm = left_qpos[:, :6].cpu().numpy() if isinstance(left_qpos, torch.Tensor) else left_qpos[:, :6]
        right_arm = right_qpos[:, :6].cpu().numpy() if isinstance(right_qpos, torch.Tensor) else right_qpos[:, :6]
        state = np.concatenate([left_arm, right_arm], axis=-1)  # (b, 12)
    else:
        # Single-arm setup
        qpos = env_unwrapped.agent.robot.get_qpos()  # (b, 7)
        state = qpos[:, :6].cpu().numpy() if isinstance(qpos, torch.Tensor) else qpos[:, :6]  # (b, 6)
    
    return state[0] if state.shape[0] == 1 else state  # Remove batch dimension if b=1


def get_camera_images(env, apply_distortion_to_front: bool = True) -> Dict[str, np.ndarray]:
    """
    Get camera images from environment.
    Returns dict with 'front', 'left_wrist', 'right_wrist' images.
    """
    env_unwrapped = env.unwrapped
    images = {}
    
    # Get front camera image
    # Try to get from sensors
    if hasattr(env_unwrapped, 'sensors') and 'front' in env_unwrapped.sensors:
        front_camera = env_unwrapped.sensors['front']
        front_img = front_camera.take_picture()
        # Convert RGBA to RGB if needed
        if front_img.shape[-1] == 4:
            front_img = front_img[..., :3]
        # Ensure uint8
        if front_img.dtype != np.uint8:
            if front_img.max() <= 1.0:
                front_img = (front_img * 255).astype(np.uint8)
            else:
                front_img = front_img.astype(np.uint8)
        
        # Apply distortion if requested
        if apply_distortion_to_front:
            front_img = apply_distortion(front_img)
        
        images['front'] = front_img
    else:
        # Fallback: render from default camera
        # This is a placeholder - actual implementation may vary
        images['front'] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Get wrist camera images
    # For now, use placeholder - will need to implement actual wrist camera rendering
    if hasattr(env_unwrapped.agent, "agents"):
        # Dual-arm: left_wrist and right_wrist
        images['left_wrist'] = np.zeros((480, 640, 3), dtype=np.uint8)
        images['right_wrist'] = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        # Single-arm: wrist
        images['wrist'] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    return images


class DataCollectionWrapper:
    """
    Wrapper to collect observations and actions during motion planning execution.
    Records data in LeRobot Dataset format.
    """
    def __init__(self, env, env_id: str, apply_distortion_to_front: bool = True):
        self.env = env
        self.env_id = env_id
        self.apply_distortion_to_front = apply_distortion_to_front
        self.task_prompt = TASK_PROMPTS.get(env_id, "")
        self.episode_data = []
        self.current_step = 0
        
    def step(self, action):
        """Wrapper around env.step to collect data."""
        # Get observation before step
        obs_before = self._get_observation()
        
        # Execute step
        obs_after, reward, terminated, truncated, info = self.env.step(action)
        
        # Record data
        data_point = {
            "observation.state": get_robot_state(self.env),
            "observation.images.front": obs_before.get("front", np.zeros((480, 640, 3), dtype=np.uint8)),
            "action": action,
            "task": self.task_prompt,
            "step": self.current_step,
        }
        
        # Add wrist camera images if available
        if "left_wrist" in obs_before:
            data_point["observation.images.left_wrist"] = obs_before["left_wrist"]
        if "right_wrist" in obs_before:
            data_point["observation.images.right_wrist"] = obs_before["right_wrist"]
        elif "wrist" in obs_before:
            data_point["observation.images.wrist"] = obs_before["wrist"]
        
        self.episode_data.append(data_point)
        self.current_step += 1
        
        return obs_after, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset and clear episode data."""
        obs, info = self.env.reset(**kwargs)
        self.episode_data = []
        self.current_step = 0
        return obs, info
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation with images."""
        images = get_camera_images(self.env, self.apply_distortion_to_front)
        return images
    
    def get_episode_data(self) -> list:
        """Get collected episode data."""
        return self.episode_data


def main():
    args = parse_args()
    
    # Create output directory for ManiSkill trajectories
    traj_output_dir = pathlib.Path(args.output_dir) / args.env_id / "motionplanning"
    traj_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment with RGB observations for data collection
    # Note: We need "rgb" or "rgbd" mode to capture images, not "none"
    env = gym.make(
        args.env_id,
        obs_mode="rgb",  # We need RGB images for LeRobot (not "none"!)
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sensor_configs=dict(shader_pack=args.shader),
        sim_backend="auto"
    )
    env = FlattenActionSpaceWrapper(env)
    
    # Wrap with RecordEpisode to record trajectories
    # This will save trajectories in ManiSkill format
    env = RecordEpisode(
        env,
        output_dir=str(traj_output_dir),
        trajectory_name=None,  # Will be auto-generated
        save_video=False,  # Don't save videos during data collection (too slow)
        source_type="motionplanning",
        source_desc="motion planning solution for Diffusion Policy training",
        video_fps=30,
        record_reward=False,
        save_on_reset=False
    )
    
    # Get solution function
    solution_func = MP_SOLUTIONS[args.env_id]
    task_prompt = TASK_PROMPTS.get(args.env_id, "")
    
    print(f"Collecting data for {args.env_id}")
    print(f"Target: {args.num_episodes} successful episodes")
    print(f"Output directory: {traj_output_dir}")
    print(f"Task prompt: {task_prompt}")
    print(f"Apply distortion: {args.apply_distortion} (will be applied during conversion)")
    print(f"Domain randomization: {args.domain_randomization}")
    print("\nNote: Trajectories are saved in ManiSkill format.")
    print("Use convert_trajectory_to_lerobot.py to convert to LeRobot Dataset format.")
    
    # Collect episodes using motion planning
    successful_episodes = 0
    total_attempts = 0
    seed = args.seed
    
    pbar = tqdm(total=args.num_episodes, desc="Collecting episodes")
    
    while successful_episodes < args.num_episodes:
        total_attempts += 1
        
        # Reset environment (RecordEpisode will start recording)
        obs, info = env.reset(seed=seed)
        seed += 1
        
        # Run motion planning solution
        # RecordEpisode will automatically record all env.step() calls
        solve_kwargs = {"seed": seed - 1, "debug": False, "vis": False}
        if args.agent_idx is not None:
            solve_kwargs["agent_idx"] = args.agent_idx
        
        try:
            result = solution_func(env, **solve_kwargs)
        except Exception as e:
            print(f"\nMotion planning failed: {e}")
            result = -1
        
        # Check if episode was successful
        if result == -1:
            env.flush_trajectory(save=False)  # Don't save failed episodes
            continue
        
        # Get final evaluation
        eval_result = env.unwrapped.evaluate()
        if not eval_result.get("success", False):
            env.flush_trajectory(save=False)  # Don't save failed episodes
            continue
        
        # Episode was successful - save trajectory
        env.flush_trajectory(save=True)
        successful_episodes += 1
        pbar.update(1)
        pbar.set_postfix({
            "success_rate": f"{successful_episodes / total_attempts * 100:.1f}%",
            "total_attempts": total_attempts
        })
    
    pbar.close()
    env.close()
    
    print(f"\nCollection complete!")
    print(f"Successful episodes: {successful_episodes}")
    print(f"Total attempts: {total_attempts}")
    print(f"Success rate: {successful_episodes / total_attempts * 100:.2f}%")
    print(f"\nTrajectories saved to: {traj_output_dir}")
    print(f"Next step: Convert trajectories to LeRobot format using:")
    print(f"  python scripts/convert_trajectory_to_lerobot.py --input-dir {traj_output_dir} --output-dir {args.output_dir}/{args.env_id} --task-name {args.env_id.lower().replace('so101-v1', '')}")


if __name__ == "__main__":
    main()
