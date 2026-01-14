"""
Evaluate trained Diffusion Policy in simulation.

This script loads a trained policy and evaluates it on the simulation environment,
measuring success rate and other metrics.
"""

import argparse
import pathlib
from typing import Dict, Any, List
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
import cv2
import os
from collections import defaultdict

try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False

# Import environments to register them
import grasp_cube.envs.tasks.lift_cube_so101
import grasp_cube.envs.tasks.stack_cube_so101
import grasp_cube.envs.tasks.sort_cube_so101

from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from grasp_cube.policies.eval_policy import DeployablePolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Diffusion Policy in simulation")
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to trained policy directory"
    )
    parser.add_argument(
        "-e", "--env-id",
        type=str,
        required=True,
        help="Environment ID (e.g., LiftCubeSO101-v1)"
    )
    parser.add_argument(
        "-n", "--num-episodes",
        type=int,
        default=50,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for inference (cuda or cpu)"
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="bi_so101",
        choices=["so101", "bi_so101"],
        help="Robot type"
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save videos of evaluation episodes"
    )
    parser.add_argument(
        "--save-failed-only",
        action="store_true",
        help="Only save videos of failed episodes"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results and videos"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed debugging information"
    )
    return parser.parse_args()


def get_observation_from_env(env) -> Dict[str, Any]:
    """
    Get observation in LeRobot format from ManiSkill environment.
    """
    env_unwrapped = env.unwrapped
    
    # Get robot state (qpos)
    if hasattr(env_unwrapped.agent, "agents"):
        # Dual-arm
        left_qpos = env_unwrapped.agent.agents[0].robot.get_qpos()
        right_qpos = env_unwrapped.agent.agents[1].robot.get_qpos()
        left_arm = left_qpos.cpu().numpy() if isinstance(left_qpos, torch.Tensor) else left_qpos
        right_arm = right_qpos.cpu().numpy() if isinstance(right_qpos, torch.Tensor) else right_qpos
        # Handle batch dimension
        if left_arm.ndim > 1:
            left_arm = left_arm[0, :6]
            right_arm = right_arm[0, :6]
        else:
            left_arm = left_arm[:6]
            right_arm = right_arm[:6]
    else:
        # Single-arm
        qpos = env_unwrapped.agent.robot.get_qpos()
        state = qpos.cpu().numpy() if isinstance(qpos, torch.Tensor) else qpos
        # Handle batch dimension
        if state.ndim > 1:
            state = state[0, :6]
        else:
            state = state[:6]
    
    # Get images
    obs_dict = env.get_obs()
    images = {}
    
    # Get front camera image
    if "image" in obs_dict and "front" in obs_dict["image"]:
        front_img = obs_dict["image"]["front"]["rgb"]
        if front_img.dtype != np.uint8:
            if front_img.max() <= 1.0:
                front_img = (front_img * 255).astype(np.uint8)
            else:
                front_img = front_img.astype(np.uint8)
        images["front"] = front_img
    else:
        images["front"] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Get wrist camera images (placeholder for now)
    if hasattr(env_unwrapped.agent, "agents"):
        images["left_wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
        images["right_wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        images["wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Get task prompt
    task_prompt = getattr(env_unwrapped, "TASK_PROMPT", "")
    
    if hasattr(env_unwrapped.agent, "agents"):
        return {
            "states": {
                "left_arm": left_arm,
                "right_arm": right_arm,
            },
            "images": images,
            "task": task_prompt,
        }
    else:
        return {
            "states": {
                "arm": state,
            },
            "images": images,
            "task": task_prompt,
        }


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    
    # Load policy
    print(f"Loading policy from {args.policy_path}")
    policy = DeployablePolicy(
        pathlib.Path(args.policy_path),
        device=args.device,
        robot_type=args.robot_type
    )
    
    # Create environment
    env = gym.make(
        args.env_id,
        obs_mode="rgb",  # Need RGB images
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="auto"
    )
    env = FlattenActionSpaceWrapper(env)
    
    print(f"Evaluating policy on {args.env_id}")
    print(f"Number of episodes: {args.num_episodes}")
    if args.save_video:
        print(f"Videos will be saved to: {video_dir}")
        if args.save_failed_only:
            print("Only failed episodes will be saved")
    
    # Evaluate
    successes = []
    episode_lengths = []
    all_rewards = []
    all_actions = []
    all_eval_results = []
    
    for episode in tqdm(range(args.num_episodes), desc="Evaluating"):
        # Reset environment and policy
        obs, info = env.reset(seed=args.seed + episode)
        policy.reset()
        
        episode_length = 0
        done = False
        episode_rewards = []
        episode_actions = []
        frames = [] if args.save_video else None
        
        while not done:
            # Get observation in LeRobot format
            observation = get_observation_from_env(env)
            
            # Get action from policy
            action = policy.get_actions(observation)
            episode_actions.append(action.copy())
            
            # Save frame if recording video
            if args.save_video:
                frame = env.render()
                if frame is not None:
                    # Convert torch.Tensor to numpy array if needed
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    # Handle batch dimension if present
                    if frame.ndim == 4 and frame.shape[0] == 1:
                        frame = frame[0]  # Remove batch dimension
                    frames.append(frame)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            done = terminated or truncated
            episode_length += 1
            
            # Note: ManiSkill environments handle truncation automatically via the TimeLimitWrapper
            # The 'truncated' flag will be True when max_episode_steps is reached
        
        # Evaluate success
        eval_result = env.unwrapped.evaluate()
        success = eval_result.get("success", False)
        successes.append(success)
        episode_lengths.append(episode_length)
        all_rewards.append(episode_rewards)
        all_actions.append(episode_actions)
        all_eval_results.append(eval_result)
        
        # Save video if requested
        if args.save_video and frames:
            should_save = not args.save_failed_only or not success
            if should_save:
                video_path = video_dir / f"episode_{episode:04d}_{'success' if success else 'failed'}.mp4"
                if len(frames) > 0:
                    # Convert all frames to numpy arrays
                    processed_frames = []
                    for frame in frames:
                        # Ensure frame is numpy array
                        if isinstance(frame, torch.Tensor):
                            frame = frame.cpu().numpy()
                        if frame.ndim == 4 and frame.shape[0] == 1:
                            frame = frame[0]
                        # Ensure uint8 dtype and correct shape
                        if frame.dtype != np.uint8:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                        # Ensure RGB format (H, W, 3)
                        if frame.shape[-1] != 3:
                            raise ValueError(f"Expected RGB image with shape (H, W, 3), got {frame.shape}")
                        processed_frames.append(frame)
                    
                    if len(processed_frames) > 0:
                        video_written = False
                        
                        # Prefer imageio as it's more reliable and compatible
                        if IMAGEIO_AVAILABLE:
                            try:
                                # imageio saves videos more reliably with better codec support
                                imageio.mimsave(
                                    str(video_path),
                                    processed_frames,
                                    fps=20,
                                    codec='libx264',
                                    quality=8,
                                    pixelformat='yuv420p'  # Better compatibility
                                )
                                video_written = True
                            except Exception as e:
                                if args.verbose:
                                    print(f"Failed to save video with imageio: {e}, trying OpenCV...")
                        
                        # Fallback to OpenCV if imageio failed or not available
                        if not video_written:
                            height, width = processed_frames[0].shape[:2]
                            # Try using H.264 codec (more compatible)
                            fourcc_options = [
                                cv2.VideoWriter_fourcc(*'avc1'),  # H.264 (most compatible)
                                cv2.VideoWriter_fourcc(*'XVID'),  # XVID
                                cv2.VideoWriter_fourcc(*'mp4v'),  # MPEG-4
                            ]
                            
                            for fourcc in fourcc_options:
                                try:
                                    out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (width, height))
                                    if out.isOpened():
                                        for frame in processed_frames:
                                            # Convert RGB to BGR for OpenCV
                                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                            out.write(frame_bgr)
                                        out.release()
                                        video_written = True
                                        break
                                except Exception as e:
                                    if args.verbose:
                                        print(f"Failed to write video with codec {fourcc}: {e}")
                                    continue
                        
                        if video_written and args.verbose:
                            print(f"Saved video: {video_path}")
                        elif not video_written:
                            print(f"Warning: Failed to save video: {video_path}")
    
    env.close()
    
    # Compute statistics
    success_rate = np.mean(successes) * 100
    avg_length = np.mean(episode_lengths)
    avg_reward = np.mean([np.sum(r) for r in all_rewards])
    
    # Action statistics
    all_actions_array = np.concatenate(all_actions, axis=0)
    action_mean = np.mean(all_actions_array, axis=0)
    action_std = np.std(all_actions_array, axis=0)
    action_min = np.min(all_actions_array, axis=0)
    action_max = np.max(all_actions_array, axis=0)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Success rate: {success_rate:.2f}% ({np.sum(successes)}/{args.num_episodes})")
    print(f"Average episode length: {avg_length:.2f} steps")
    print(f"Average total reward: {avg_reward:.4f}")
    print(f"Target: >50% success rate for full points")
    
    # Detailed statistics
    if args.verbose:
        print(f"\nDetailed Statistics:")
        print(f"  Episode lengths: min={np.min(episode_lengths)}, max={np.max(episode_lengths)}, std={np.std(episode_lengths):.2f}")
        print(f"  Total rewards: min={np.min([np.sum(r) for r in all_rewards]):.4f}, max={np.max([np.sum(r) for r in all_rewards]):.4f}")
        
        # Eval result statistics
        eval_keys = set()
        for er in all_eval_results:
            eval_keys.update(er.keys())
        for key in eval_keys:
            values = [er.get(key, None) for er in all_eval_results]
            if all(v is not None and isinstance(v, (int, float, bool, np.number)) for v in values):
                values_array = np.array([float(v) for v in values])
                print(f"  {key}: mean={np.mean(values_array):.4f}, std={np.std(values_array):.4f}")
        
        print(f"\nAction Statistics:")
        print(f"  Mean: {action_mean}")
        print(f"  Std:  {action_std}")
        print(f"  Min:  {action_min}")
        print(f"  Max:  {action_max}")
    
    # Success criteria
    if success_rate >= 50:
        print("\n✓ Success rate meets requirement (>50%)")
    elif success_rate >= 20:
        print("\n⚠ Success rate is 20-50% (partial points)")
    else:
        print("\n✗ Success rate is below 20% (no points)")
    
    # Save summary to file
    summary_file = output_dir / "evaluation_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"{'='*60}\n")
        f.write(f"Policy path: {args.policy_path}\n")
        f.write(f"Environment: {args.env_id}\n")
        f.write(f"Number of episodes: {args.num_episodes}\n")
        f.write(f"Success rate: {success_rate:.2f}% ({np.sum(successes)}/{args.num_episodes})\n")
        f.write(f"Average episode length: {avg_length:.2f} steps\n")
        f.write(f"Average total reward: {avg_reward:.4f}\n")
        if args.verbose:
            f.write(f"\nAction Statistics:\n")
            f.write(f"  Mean: {action_mean}\n")
            f.write(f"  Std:  {action_std}\n")
            f.write(f"  Min:  {action_min}\n")
            f.write(f"  Max:  {action_max}\n")
    
    print(f"\nSummary saved to: {summary_file}")
    if args.save_video:
        print(f"Videos saved to: {video_dir}")


if __name__ == "__main__":
    main()
