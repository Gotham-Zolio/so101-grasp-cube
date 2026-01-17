"""
Evaluate trained ACT policy in simulation.

This script loads a trained ACT policy and evaluates it on the simulation environment,
measuring success rate and other metrics.
"""

import argparse
import pathlib
from typing import Dict, Any, List, Optional
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

# Import data augmentation utilities
from grasp_cube.utils.data_augmentation import apply_visual_disturbance

# LeRobot imports
try:
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Error: LeRobot not installed. Please install it: pip install lerobot")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ACT Policy in simulation")
    parser.add_argument(
        "--policy-path",
        type=str,
        required=True,
        help="Path to trained policy directory (LeRobot output_dir or checkpoint path)"
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
        "--save-camera",
        action="store_true",
        help="Save camera images/videos separately"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed debugging information"
    )
    parser.add_argument(
        "--visual-disturbance",
        action="store_true",
        help="Apply visual disturbance during evaluation (for generalization testing)"
    )
    parser.add_argument(
        "--brightness-factor",
        type=float,
        default=1.0,
        help="Brightness multiplier for visual disturbance (>1.0 = brighter, <1.0 = darker)"
    )
    parser.add_argument(
        "--contrast-factor",
        type=float,
        default=1.0,
        help="Contrast multiplier for visual disturbance (>1.0 = more contrast, <1.0 = less contrast)"
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Standard deviation of Gaussian noise for visual disturbance (0.0 = no noise)"
    )
    return parser.parse_args()


def load_policy(policy_path: pathlib.Path, device: str = "cuda"):
    """Load ACT policy from checkpoint."""
    policy_path = policy_path.resolve()
    
    # Handle LeRobot native training checkpoint structure
    # LeRobot saves checkpoints as: output_dir/checkpoints/STEP/pretrained_model/
    # Note: Checkpoint directories use zero-padded names (e.g., "020000", "040000") or "last"
    if policy_path.is_dir():
        checkpoints_dir = policy_path / "checkpoints"
        if checkpoints_dir.exists() and checkpoints_dir.is_dir():
            # Find the latest checkpoint step
            # Preserve original directory names to maintain zero-padding (e.g., "020000")
            checkpoint_dirs = []
            for item in checkpoints_dir.iterdir():
                if item.is_dir():
                    if item.name == "last":
                        # "last" is a special checkpoint pointing to the latest
                        checkpoint_dirs.append((float('inf'), item.name))
                    elif item.name.isdigit():
                        # Convert to int for comparison, but keep original name for path
                        checkpoint_dirs.append((int(item.name), item.name))
            
            if checkpoint_dirs:
                # Sort by step number (int) and get the latest
                checkpoint_dirs.sort(key=lambda x: x[0])
                latest_step_name = checkpoint_dirs[-1][1]  # Use original name (preserves zero-padding)
                pretrained_model_dir = checkpoints_dir / latest_step_name / "pretrained_model"
                if pretrained_model_dir.exists():
                    print(f"Found LeRobot checkpoint structure. Using latest checkpoint: {pretrained_model_dir}")
                    policy_path = pretrained_model_dir
                else:
                    raise FileNotFoundError(
                        f"Checkpoint directory found but pretrained_model not found. "
                        f"Expected: {pretrained_model_dir}"
                    )
            else:
                raise FileNotFoundError(
                    f"No checkpoint steps found in {checkpoints_dir}. "
                    f"Expected directories named with step numbers (e.g., 020000, 040000) or 'last'."
                )
    
    if not policy_path.exists():
        raise FileNotFoundError(f"Policy path does not exist: {policy_path}")
    
    # Load policy configuration
    policy_config = PreTrainedConfig.from_pretrained(
        str(policy_path),
        local_files_only=True
    )
    
    if not isinstance(policy_config, ACTConfig):
        raise ValueError(
            f"Expected ACTConfig, got {type(policy_config)}. "
            f"This script only supports ACT policies."
        )
    
    print(f"Loading ACT policy from {policy_path}")
    print(f"  Policy type: ACT")
    print(f"  Chunk size: {policy_config.chunk_size}")
    print(f"  N action steps: {policy_config.n_action_steps}")
    print(f"  N obs steps: {policy_config.n_obs_steps}")
    
    # Load policy model
    policy = ACTPolicy.from_pretrained(
        str(policy_path),
        config=policy_config,
        local_files_only=True
    )
    
    policy.to(device)
    policy.eval()
    
    # Print input features
    if hasattr(policy_config, 'input_features'):
        input_features = getattr(policy_config, 'input_features', None)
        if input_features is not None and isinstance(input_features, dict):
            feature_keys = [str(k) for k in input_features.keys()]
            print(f"  Input features: {feature_keys}")
    
    # Check normalization statistics
    print(f"  Normalization mode: {policy_config.normalization_mapping}")
    
    # Verify normalization buffers are loaded (not infinity or all zeros)
    # This is critical - if stats are infinity or all zeros, normalization won't work correctly
    normalization_ok = True
    if hasattr(policy, 'normalize_inputs'):
        for key in policy_config.input_features.keys():
            buffer_name = "buffer_" + key.replace(".", "_")
            if hasattr(policy.normalize_inputs, buffer_name):
                buffer = getattr(policy.normalize_inputs, buffer_name)
                if "mean" in buffer and "std" in buffer:
                    mean = buffer["mean"]
                    std = buffer["std"]
                    if torch.isinf(mean).any() or torch.isinf(std).any():
                        print(f"  ERROR: Normalization stats for {key} are infinity! Model will not work correctly.")
                        print(f"    This usually means the model was not properly trained or checkpoint is corrupted.")
                        normalization_ok = False
                    else:
                        mean_val = mean.squeeze().cpu().numpy()
                        std_val = std.squeeze().cpu().numpy()
                        
                        # Check if std is all zeros (critical issue for normalization)
                        if np.allclose(std_val, 0.0):
                            print(f"  ⚠️  WARNING: Normalization std for {key} is all zeros!")
                            print(f"    This will cause division by zero during normalization.")
                            print(f"    Attempting to fix by computing stats from actual state values...")
                            
                            # Compute approximate std from typical state ranges
                            # For joint positions, typical range is roughly [-2, 2] radians
                            # Use a reasonable std estimate based on typical joint position variance
                            if key == "observation.state":
                                # For 6-DOF arm, typical std for joint positions is around 0.5-1.0
                                # Use a conservative estimate
                                estimated_std = torch.ones_like(std) * 1.0  # Assume std=1.0 for joint positions
                                buffer["std"] = estimated_std.to(device)
                                print(f"    ✓ Fixed: Set std to 1.0 (identity normalization for state)")
                                print(f"    Note: State normalization will be effectively disabled (x - mean) / 1.0")
                                print(f"    This is acceptable if state values are already in a reasonable range")
                            else:
                                # For other features, use identity normalization
                                buffer["std"] = torch.ones_like(std).to(device)
                                print(f"    ✓ Fixed: Set std to 1.0 (identity normalization)")
                        
                        # For images, mean/std are per-channel (C, 1, 1)
                        if key.startswith("observation.images."):
                            print(f"  Normalization stats for {key}: mean per channel={mean_val.flatten()}, std per channel={std_val.flatten()}")
                        else:
                            print(f"  Normalization stats for {key}: mean={mean_val}, std={std_val}")
                elif "min" in buffer and "max" in buffer:
                    min_val = buffer["min"]
                    max_val = buffer["max"]
                    if torch.isinf(min_val).any() or torch.isinf(max_val).any():
                        print(f"  ERROR: Normalization stats for {key} are infinity!")
                        normalization_ok = False
                    else:
                        print(f"  Normalization stats for {key}: min={min_val.squeeze().cpu().numpy()}, max={max_val.squeeze().cpu().numpy()}")
    
    if not normalization_ok:
        raise RuntimeError(
            "Normalization statistics are not properly loaded. "
            "The model checkpoint may be corrupted or incomplete. "
            "Please check the checkpoint or retrain the model."
        )
    
    return policy, policy_config


def determine_cameras_from_env(env_id: str) -> List[str]:
    """Determine which cameras to use based on environment ID.
    
    All tasks (lift, stack, sort) use the same three cameras: front + left_side + right_side.
    """
    return ["front", "left_side", "right_side"]


def get_observation_from_env(
    env, 
    required_cameras: List[str], 
    verbose: bool = False,
    apply_visual_disturbance_flag: bool = False,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0,
    noise_std: float = 0.0,
) -> tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """
    Get observation in LeRobot format from ManiSkill environment.
    
    Args:
        env: ManiSkill environment
        required_cameras: List of camera names to extract (e.g., ["front", "left_side", "right_side"])
        verbose: Print debug information
        apply_visual_disturbance_flag: Whether to apply visual disturbance
        brightness_factor: Brightness multiplier for visual disturbance
        contrast_factor: Contrast multiplier for visual disturbance
        noise_std: Standard deviation of Gaussian noise for visual disturbance
    
    Returns:
        Tuple of (observation_dict, raw_images_dict):
        - observation_dict: Dict with "observation.state" and "observation.images.*" keys (for policy)
        - raw_images_dict: Dict with original images (for video saving)
    """
    env_unwrapped = env.unwrapped
    
    # Ensure scene is updated before taking pictures
    if hasattr(env_unwrapped, 'scene'):
        env_unwrapped.scene.update_render()
    
    # Get robot state (qpos)
    num_agents = len(env_unwrapped.agent.agents) if hasattr(env_unwrapped.agent, "agents") else 1
    
    if num_agents > 1:
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
        state = np.concatenate([left_arm, right_arm], axis=-1)  # (12,)
    else:
        # Single-arm
        qpos = env_unwrapped.agent.agents[0].robot.get_qpos() if hasattr(env_unwrapped.agent, "agents") else env_unwrapped.agent.robot.get_qpos()
        state = qpos.cpu().numpy() if isinstance(qpos, torch.Tensor) else qpos
        # Handle batch dimension
        if state.ndim > 1:
            state = state[0, :6]
        else:
            state = state[:6]  # (6,)
    
    # Get images from sensors
    images = {}
    obs_dict = env.get_obs()
    
    # All tasks use the same three cameras: front + left_side + right_side (fixed pose cameras)
    for camera_name in required_cameras:
        camera_img = None
        
        # Get camera from environment sensor configs (all cameras are fixed pose, no mounting needed)
        if hasattr(env_unwrapped, 'sensors') and camera_name in env_unwrapped.sensors:
            try:
                camera = env_unwrapped.sensors[camera_name]
                camera_img = camera.take_picture()
                if isinstance(camera_img, torch.Tensor):
                    camera_img = camera_img.cpu().numpy()
                if camera_img.ndim == 4:
                    camera_img = camera_img[0]  # Remove batch dimension
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to get {camera_name} from sensors: {e}")
        
        # Fallback: try from sensor_data
        if camera_img is None and "sensor_data" in obs_dict:
            sensor_data = obs_dict["sensor_data"]
            if isinstance(sensor_data, dict) and camera_name in sensor_data:
                camera_data = sensor_data[camera_name]
                if isinstance(camera_data, dict) and "rgb" in camera_data:
                    camera_img = camera_data["rgb"]
                elif isinstance(camera_data, (np.ndarray, torch.Tensor)):
                    camera_img = camera_data
        
        # Process image
        if camera_img is not None:
            # Convert torch.Tensor to numpy
            if isinstance(camera_img, torch.Tensor):
                camera_img = camera_img.cpu().numpy()
            
            # Handle batch dimension
            if camera_img.ndim == 4:
                camera_img = camera_img[0]
            elif camera_img.ndim == 3 and camera_img.shape[0] in [3, 4]:
                # (C, H, W) -> (H, W, C)
                camera_img = camera_img.transpose(1, 2, 0)
            
            # Convert RGBA to RGB
            if camera_img.ndim == 3 and camera_img.shape[-1] == 4:
                camera_img = camera_img[..., :3]
            
            # Convert to uint8
            if camera_img.dtype != np.uint8:
                if camera_img.max() <= 1.0:
                    camera_img = (camera_img * 255).astype(np.uint8)
                else:
                    camera_img = np.clip(camera_img, 0, 255).astype(np.uint8)
            
            # Ensure correct shape (H, W, C)
            if camera_img.shape[:2] != (480, 640):
                camera_img = cv2.resize(camera_img, (640, 480))
            
            images[camera_name] = camera_img
        else:
            # Use black placeholder if camera not available
            if verbose:
                print(f"Warning: {camera_name} camera not found, using black placeholder")
            images[camera_name] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Apply visual disturbance if requested (for generalization testing)
    raw_images = images.copy()  # Keep original images for video saving
    if apply_visual_disturbance_flag:
        images = apply_visual_disturbance(
            images,
            brightness_factor=brightness_factor,
            contrast_factor=contrast_factor,
            noise_std=noise_std,
        )
    
    # Build observation dict in LeRobot format
    # State should be raw joint positions (float32), will be normalized by policy
    observation = {
        "observation.state": state.astype(np.float32),
    }
    
    # Debug: verify state dimension matches expected
    if verbose:
        print(f"  State: shape={state.shape}, dtype={state.dtype}, range=[{state.min():.3f}, {state.max():.3f}]")
    
    # Add images (convert to tensor format: C, H, W)
    # IMPORTANT: Images should be in [0, 1] range (float32) for LeRobot normalization
    # LeRobot's normalize_inputs will apply MEAN_STD normalization automatically
    for camera_name, img in images.items():
        # Convert (H, W, C) to (C, H, W) and normalize to [0, 1]
        # img is already uint8 [0, 255], convert to float [0, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        observation[f"observation.images.{camera_name}"] = img_tensor
    
    return observation, raw_images
    
    return observation, raw_images  # Return both for video saving


def prepare_batch_for_policy(observation: Dict[str, Any], device: str = "cuda", verbose: bool = False) -> Dict[str, torch.Tensor]:
    """
    Prepare observation batch for ACT policy.
    
    ACT expects:
    - observation.state: (batch_size, state_dim) - float32, raw values (will be normalized by policy)
    - observation.images.*: (batch_size, C, H, W) - float32, [0, 1] range (will be normalized by policy)
    
    Note: The policy's normalize_inputs will automatically apply MEAN_STD normalization.
    We just need to provide data in the correct format and range.
    """
    batch = {}
    
    # State: (state_dim,) -> (1, state_dim)
    if "observation.state" in observation:
        state = observation["observation.state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_dim)
        batch["observation.state"] = state.to(device)
        
        if verbose:
            print(f"  State: shape={state.shape}, range=[{state.min():.3f}, {state.max():.3f}]")
    
    # Images: (C, H, W) -> (1, C, H, W)
    # Images should be in [0, 1] range (float32)
    for key, value in observation.items():
        if key.startswith("observation.images."):
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value).float()
            if value.dim() == 3:
                value = value.unsqueeze(0)  # (1, C, H, W)
            
            # Ensure values are in [0, 1] range
            if value.max() > 1.0:
                print(f"  WARNING: {key} has values > 1.0 (max={value.max():.3f}), normalizing to [0, 1]")
                value = value / 255.0
            
            batch[key] = value.to(device)
            
            if verbose:
                print(f"  {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}], dtype={value.dtype}")
    
    return batch


def main():
    args = parse_args()
    
    if not LEROBOT_AVAILABLE:
        raise ImportError("LeRobot is required. Please install it: pip install lerobot")
    
    # Create output directory structure: output_dir/env_id/
    # This allows each task to have its own folder
    base_output_dir = pathlib.Path(args.output_dir)
    task_output_dir = base_output_dir / args.env_id
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for videos
    video_dir = task_output_dir / "videos"
    camera_dir = task_output_dir / "camera_videos"
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)
    if args.save_camera:
        camera_dir.mkdir(parents=True, exist_ok=True)
    
    # Use task_output_dir as the main output directory
    output_dir = task_output_dir
    
    # Load policy
    print(f"Loading policy from {args.policy_path}")
    policy, policy_config = load_policy(pathlib.Path(args.policy_path), device=args.device)
    
    # Determine required cameras based on environment
    required_cameras = determine_cameras_from_env(args.env_id)
    print(f"Required cameras: {required_cameras}")
    print(f"Output directory: {output_dir}")
    
    # Create environment
    env = gym.make(
        args.env_id,
        obs_mode="rgb",  # Need RGB images
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="auto"
    )
    # Only apply FlattenActionSpaceWrapper if action space is Dict (multi-agent/dual-arm)
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    
    # Auto-detect robot type from environment
    env_unwrapped = env.unwrapped
    num_agents = len(env_unwrapped.agent.agents) if hasattr(env_unwrapped.agent, "agents") else 1
    robot_type = "bi_so101" if num_agents > 1 else "so101"
    
    print(f"Evaluating policy on {args.env_id}")
    print(f"Number of episodes: {args.num_episodes}")
    print(f"Robot type: {robot_type} ({num_agents} arm{'s' if num_agents > 1 else ''})")
    if args.visual_disturbance:
        print(f"Visual disturbance enabled:")
        print(f"  - Brightness factor: {args.brightness_factor}")
        print(f"  - Contrast factor: {args.contrast_factor}")
        print(f"  - Noise std: {args.noise_std}")
    if args.save_video:
        print(f"Videos will be saved to: {video_dir}")
        if args.save_failed_only:
            print("Only failed episodes will be saved")
    if args.save_camera:
        print(f"Camera videos will be saved to: {camera_dir}")
        if args.save_failed_only:
            print("Only failed episodes' camera videos will be saved")
    
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
        camera_frames = {cam: [] for cam in required_cameras} if args.save_camera else None
        
        while not done:
            # Render scene before getting observations
            if args.save_video:
                _ = env.render()
            
            # Get observation in LeRobot format
            observation, raw_images = get_observation_from_env(
                env, 
                required_cameras=required_cameras,
                verbose=args.verbose and episode == 0,  # Only verbose for first episode
                apply_visual_disturbance_flag=args.visual_disturbance,
                brightness_factor=args.brightness_factor,
                contrast_factor=args.contrast_factor,
                noise_std=args.noise_std,
            )
            
            # Save camera frames if requested
            if args.save_camera and camera_frames is not None:
                for camera_name in required_cameras:
                    if camera_name in raw_images:
                        camera_frames[camera_name].append(raw_images[camera_name].copy())
            
            # Prepare batch for policy
            batch = prepare_batch_for_policy(
                observation, 
                device=args.device,
                verbose=args.verbose and episode == 0 and len(episode_actions) == 0  # Only first step of first episode
            )
            
            # Get action from policy
            # select_action automatically handles normalization and returns unnormalized action
            with torch.no_grad():
                action = policy.select_action(batch)  # Returns (action_dim,) - already unnormalized
            
            # Convert to numpy
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            else:
                action = np.asarray(action)
            
            # Ensure action is 1D
            if action.ndim > 1:
                action = action.flatten()
            
            # Debug: print action statistics (only first step of first episode)
            if args.verbose and episode == 0 and len(episode_actions) == 0:
                print(f"  Action: shape={action.shape}, range=[{action.min():.3f}, {action.max():.3f}], mean={action.mean():.3f}")
            
            # Clip action to action space bounds (safety check)
            if hasattr(env, 'action_space'):
                if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                    action_before_clip = action.copy()
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                    if args.verbose and episode == 0 and len(episode_actions) == 0 and not np.allclose(action, action_before_clip):
                        print(f"  WARNING: Action was clipped! Before: {action_before_clip}, After: {action}")
            
            episode_actions.append(action.copy())
            
            # Save frame if recording video
            if args.save_video:
                frame = env.render()
                if frame is not None:
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    if frame.ndim == 4 and frame.shape[0] == 1:
                        frame = frame[0]
                    frames.append(frame)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_rewards.append(reward)
            done = terminated or truncated
            episode_length += 1
        
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
                    processed_frames = []
                    for frame in frames:
                        if isinstance(frame, torch.Tensor):
                            frame = frame.cpu().numpy()
                        if frame.ndim == 4 and frame.shape[0] == 1:
                            frame = frame[0]
                        if frame.dtype != np.uint8:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                        if frame.shape[-1] == 3:
                            processed_frames.append(frame)
                    
                    if len(processed_frames) > 0:
                        video_written = False
                        if IMAGEIO_AVAILABLE:
                            try:
                                imageio.mimsave(
                                    str(video_path),
                                    processed_frames,
                                    fps=20,
                                    codec='libx264',
                                    quality=8,
                                    pixelformat='yuv420p'
                                )
                                video_written = True
                            except Exception as e:
                                if args.verbose:
                                    print(f"Failed to save video with imageio: {e}")
                        
                        if not video_written:
                            height, width = processed_frames[0].shape[:2]
                            fourcc_options = [
                                cv2.VideoWriter_fourcc(*'avc1'),
                                cv2.VideoWriter_fourcc(*'XVID'),
                                cv2.VideoWriter_fourcc(*'mp4v'),
                            ]
                            for fourcc in fourcc_options:
                                try:
                                    out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (width, height))
                                    if out.isOpened():
                                        for frame in processed_frames:
                                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                            out.write(frame_bgr)
                                        out.release()
                                        video_written = True
                                        break
                                except Exception:
                                    continue
                        
                        if video_written and args.verbose:
                            print(f"Saved video: {video_path}")
        
        # Save camera videos if requested
        if args.save_camera and camera_frames:
            should_save = not args.save_failed_only or not success
            if should_save:
                for camera_name, frames in camera_frames.items():
                    if len(frames) == 0:
                        continue
                    camera_video_path = camera_dir / f"episode_{episode:04d}_{camera_name}_{'success' if success else 'failed'}.mp4"
                    
                    processed_frames = []
                    for frame in frames:
                        if isinstance(frame, torch.Tensor):
                            frame = frame.cpu().numpy()
                        if frame.ndim == 4 and frame.shape[0] == 1:
                            frame = frame[0]
                        if frame.dtype != np.uint8:
                            frame = np.clip(frame, 0, 255).astype(np.uint8)
                        if frame.shape[-1] == 3:
                            processed_frames.append(frame)
                    
                    if len(processed_frames) > 0:
                        video_written = False
                        if IMAGEIO_AVAILABLE:
                            try:
                                imageio.mimsave(
                                    str(camera_video_path),
                                    processed_frames,
                                    fps=20,
                                    codec='libx264',
                                    quality=8,
                                    pixelformat='yuv420p'
                                )
                                video_written = True
                            except Exception:
                                pass
                        
                        if not video_written:
                            height, width = processed_frames[0].shape[:2]
                            fourcc_options = [
                                cv2.VideoWriter_fourcc(*'avc1'),
                                cv2.VideoWriter_fourcc(*'XVID'),
                                cv2.VideoWriter_fourcc(*'mp4v'),
                            ]
                            for fourcc in fourcc_options:
                                try:
                                    out = cv2.VideoWriter(str(camera_video_path), fourcc, 20.0, (width, height))
                                    if out.isOpened():
                                        for frame in processed_frames:
                                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                            out.write(frame_bgr)
                                        out.release()
                                        video_written = True
                                        break
                                except Exception:
                                    continue
    
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
    if args.save_camera:
        print(f"Camera videos saved to: {camera_dir}")


if __name__ == "__main__":
    main()
