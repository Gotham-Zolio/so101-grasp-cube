"""
Data collection script using motion planning solutions.
Collects trajectories for Diffusion Policy training.

This script:
1. Uses motion planning solutions to generate successful trajectories
2. Records trajectories in ManiSkill format (using RecordEpisode wrapper)
3. Trajectories can then be converted to LeRobot Dataset format using convert_trajectory_to_lerobot.py

Key features:
- Records joint positions (not end-effector poses) as actions
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
from grasp_cube.motionplanning.so101.solutions.avoid_obstacle import solve as solveAvoidObstacle


# Task prompts
TASK_PROMPTS = {
    "LiftCubeSO101-v1": "Pick up the red cube and lift it.",
    "StackCubeSO101-v1": "Stack the red cube on top of the green cube.",
    "SortCubeSO101-v1": "Move the red cube to the left region and the green cube to the right region.",
    "AvoidObstacleSO101-v1": "Lift the red cube over obstacles to the middle. Lift the green cube over obstacles to the middle.",
}

# Motion planning solutions
MP_SOLUTIONS = {
    "LiftCubeSO101-v1": solveLiftCube,
    "StackCubeSO101-v1": solveStackCube,
    "SortCubeSO101-v1": solveSortCube,
    "AvoidObstacleSO101-v1": solveAvoidObstacle,
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


def get_camera_images(env) -> Dict[str, np.ndarray]:
    """
    Get camera images from environment.
    Returns dict with 'front', 'left_side', 'right_side' images.
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
        
        images['front'] = front_img
    else:
        # Fallback: render from default camera
        # This is a placeholder - actual implementation may vary
        images['front'] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Get side camera images (left_side and right_side) - fixed pose cameras
    # All tasks (lift, stack, sort) use the same three cameras: front + left_side + right_side
    if hasattr(env_unwrapped, 'sensors'):
        # Left side camera
        if 'left_side' in env_unwrapped.sensors:
            try:
                left_side_img = _get_image_from_sensor(env_unwrapped.sensors['left_side'])
                images['left_side'] = left_side_img
            except Exception as e:
                print(f"Warning: Failed to get left_side from sensors: {e}")
                images['left_side'] = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            images['left_side'] = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Right side camera
        if 'right_side' in env_unwrapped.sensors:
            try:
                right_side_img = _get_image_from_sensor(env_unwrapped.sensors['right_side'])
                images['right_side'] = right_side_img
            except Exception as e:
                print(f"Warning: Failed to get right_side from sensors: {e}")
                images['right_side'] = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            images['right_side'] = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        # Fallback: use placeholders if sensors not available
        images['left_side'] = np.zeros((480, 640, 3), dtype=np.uint8)
        images['right_side'] = np.zeros((480, 640, 3), dtype=np.uint8)


def _get_image_from_sensor(sensor):
    """Helper function to get image from sensor and convert to uint8 RGB."""
    img = sensor.take_picture()
    if img.shape[-1] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    return img
    
    return images


class DataCollectionWrapper:
    """
    Wrapper to collect observations and actions during motion planning execution.
    Records data in LeRobot Dataset format.
    """
    def __init__(self, env, env_id: str):
        self.env = env
        self.env_id = env_id
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
        
        # Add side camera images (all tasks use front + left_side + right_side)
        if "left_side" in obs_before:
            data_point["observation.images.left_side"] = obs_before["left_side"]
        if "right_side" in obs_before:
            data_point["observation.images.right_side"] = obs_before["right_side"]
        
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
        images = get_camera_images(self.env)
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
    # Only wrap with FlattenActionSpaceWrapper if action space is Dict (multi-agent)
    # Single-arm environments have Box action space which doesn't need flattening
    from gymnasium.spaces import Dict
    if isinstance(env.action_space, Dict):
        env = FlattenActionSpaceWrapper(env)
    
    # Wrap with RecordEpisode to record trajectories
    # This will save trajectories in ManiSkill format
    # IMPORTANT: RecordEpisode should automatically save get_obs() data when obs_mode="rgb"
    # But we need to ensure obs_mode is set correctly and sensors are available
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
    
    # Debug: Verify environment has sensors configured
    env_unwrapped = env.unwrapped
    print(f"Environment obs_mode: {getattr(env_unwrapped, 'obs_mode', 'unknown')}")
    print(f"Environment has sensors attribute: {hasattr(env_unwrapped, 'sensors')}")
    if hasattr(env_unwrapped, 'sensors'):
        print(f"Available sensors: {list(env_unwrapped.sensors.keys()) if isinstance(env_unwrapped.sensors, dict) else 'not a dict'}")
    
    # Test: Get one observation to verify sensor_data is available
    # IMPORTANT: For dual-arm tasks, ensure cameras are mounted after reset
    test_obs, _ = env.reset(seed=42)
    
    # After reset, verify cameras are available
    # Note: All tasks now use fixed-pose cameras (front, left_side, right_side), no mounting needed
    env_unwrapped = env.unwrapped
    
    test_obs_dict = env.get_obs()
    print(f"Test obs_dict keys: {list(test_obs_dict.keys())}")
    if "sensor_data" in test_obs_dict:
        print(f"✓ sensor_data is available in get_obs()")
        if isinstance(test_obs_dict["sensor_data"], dict):
            sensor_keys = list(test_obs_dict['sensor_data'].keys())
            print(f"  sensor_data keys: {sensor_keys}")
            # Verify all three cameras (front, left_side, right_side) are present
            expected_cameras = ['front', 'left_side', 'right_side']
            missing_cameras = [cam for cam in expected_cameras if cam not in sensor_keys]
            if not missing_cameras:
                print(f"  ✓ All cameras are available: {expected_cameras}")
            else:
                print(f"  ⚠️  WARNING: Missing cameras in sensor_data!")
                print(f"     Expected: {expected_cameras}")
                print(f"     Missing: {missing_cameras}")
                print(f"     Got: {sensor_keys}")
    else:
        print(f"⚠️  WARNING: sensor_data NOT in get_obs()!")
        print(f"  This means RecordEpisode will NOT save images!")
        print(f"  Check environment configuration (obs_mode='rgb' should include sensor_data)")
    
    # Get solution function
    solution_func = MP_SOLUTIONS[args.env_id]
    task_prompt = TASK_PROMPTS.get(args.env_id, "")
    
    print(f"Collecting data for {args.env_id}")
    print(f"Target: {args.num_episodes} successful episodes")
    print(f"Output directory: {traj_output_dir}")
    print(f"Task prompt: {task_prompt}")
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
