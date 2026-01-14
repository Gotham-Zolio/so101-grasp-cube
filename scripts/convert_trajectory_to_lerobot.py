"""
Convert ManiSkill trajectory (h5 format) to LeRobot Dataset format.

This script reads ManiSkill trajectory files and converts them to LeRobot Dataset format,
applying camera distortion and formatting observations/actions correctly.
"""

import argparse
import h5py
import numpy as np
import pathlib
from tqdm import tqdm
import cv2
import json
from typing import Dict, Any, List, Optional

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Warning: LeRobot not installed. Will save in intermediate format.")

from grasp_cube.utils.image_distortion import apply_distortion


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ManiSkill trajectories to LeRobot Dataset")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing ManiSkill trajectory h5 files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for LeRobot Dataset"
    )
    parser.add_argument(
        "--task-name",
        type=str,
        required=True,
        help="Task name (e.g., lift, stack, sort)"
    )
    parser.add_argument(
        "--apply-distortion",
        action="store_true",
        default=True,
        help="Apply camera distortion to front camera images"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed conversion information"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect h5 file structure without converting"
    )
    return parser.parse_args()


def inspect_h5_structure(h5_path: pathlib.Path):
    """Inspect the structure of a ManiSkill h5 file for debugging."""
    with h5py.File(h5_path, 'r') as f:
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{indent}{name}/: group")
        
        print(f"Structure of {h5_path}:")
        f.visititems(print_structure)


def read_maniskill_trajectory(h5_path: pathlib.Path, verbose: bool = False, traj_index: int = 0) -> Dict[str, Any]:
    """
    Read ManiSkill trajectory h5 file.
    Returns dict with observations, actions, and metadata.
    
    ManiSkill RecordEpisode saves data in format:
    - traj_XX/actions: (T, action_dim)
    - traj_XX/obs/agent/so101-0/qpos: (T+1, 6) for left arm
    - traj_XX/obs/agent/so101-1/qpos: (T+1, 6) for right arm
    - traj_XX/obs/sensor_data/front/rgb: (T+1, H, W, 3)
    """
    data = {}
    
    try:
        f = h5py.File(h5_path, 'r')
    except (RuntimeError, OSError, IOError) as e:
        raise RuntimeError(f"Failed to open h5 file {h5_path}: {e}. File may be corrupted.") from e
    
    try:
        if verbose:
            try:
                top_keys = list(f.keys())
                print(f"    Top-level keys: {top_keys}")
            except (RuntimeError, OSError) as e:
                raise RuntimeError(f"Failed to read keys from {h5_path}: {e}. File may be corrupted.") from e
        
        # Find trajectory group (could be "trajectory" or "traj_XX")
        # If multiple trajectories exist, process the one at traj_index
        traj_group = None
        traj_group_name = None
        
        # Collect all trajectory groups
        try:
            traj_keys = [key for key in f.keys() if key.startswith("traj")]
        except (RuntimeError, OSError) as e:
            raise RuntimeError(f"Failed to iterate keys in {h5_path}: {e}. File may be corrupted.") from e
        traj_keys.sort()  # Sort to ensure consistent ordering
        
        if len(traj_keys) > 0:
            # Use the specified trajectory index
            if traj_index < len(traj_keys):
                traj_group_name = traj_keys[traj_index]
                traj_group = f[traj_group_name]
                if verbose:
                    print(f"    Processing trajectory {traj_index}/{len(traj_keys)}: {traj_group_name}")
            else:
                raise ValueError(f"Trajectory index {traj_index} out of range. Found {len(traj_keys)} trajectories.")
        
        if traj_group is None and "trajectory" in f:
            traj_group = f["trajectory"]
            traj_group_name = "trajectory"
            if verbose:
                print(f"    Found trajectory group: trajectory")
        
        if traj_group is not None:
            if verbose:
                print(f"    Trajectory group keys: {list(traj_group.keys())}")
            
            # Read actions - try different possible locations
            action_found = False
            for action_key in ["actions", "action"]:
                if action_key in traj_group:
                    data["action"] = traj_group[action_key][:]
                    action_found = True
                    if verbose:
                        print(f"    Found actions at '{action_key}': shape={data['action'].shape}")
                    break
            
            if not action_found:
                # Try direct access
                if "action" in f or "actions" in f:
                    action_key = "actions" if "actions" in f else "action"
                    data["action"] = f[action_key][:]
                    action_found = True
                    if verbose:
                        print(f"    Found actions at top level: shape={data['action'].shape}")
            
            if not action_found:
                # List all available keys for debugging
                if verbose:
                    print(f"    Trajectory group keys: {list(traj_group.keys())}")
                raise ValueError(f"No 'actions' or 'action' found. Available keys in trajectory group: {list(traj_group.keys())}")
            
            # Read observations
            # Structure: traj_XX/obs/agent/so101-0/qpos, traj_XX/obs/sensor_data/front/rgb
            if "obs" in traj_group:
                obs_group = traj_group["obs"]
                data["obs"] = {}
                
                # Extract agent qpos (robot state)
                if "agent" in obs_group:
                    agent_group = obs_group["agent"]
                    # For dual-arm: so101-0 and so101-1
                    # For single-arm: so101-0 or just qpos
                    qpos_list = []
                    for agent_key in sorted(agent_group.keys()):
                        agent_data = agent_group[agent_key]
                        if isinstance(agent_data, h5py.Group) and "qpos" in agent_data:
                            qpos = agent_data["qpos"][:]
                            qpos_list.append(qpos)
                            if verbose:
                                print(f"    Found qpos for {agent_key}: shape={qpos.shape}")
                    
                    if len(qpos_list) > 0:
                        if len(qpos_list) == 1:
                            data["obs"]["agent_qpos"] = qpos_list[0]
                        else:
                            # Concatenate dual-arm qpos
                            data["obs"]["agent_qpos"] = np.concatenate(qpos_list, axis=-1)
                
                # Extract sensor data (images)
                if "sensor_data" in obs_group:
                    sensor_group = obs_group["sensor_data"]
                    for sensor_name in sensor_group.keys():
                        sensor_data = sensor_group[sensor_name]
                        if isinstance(sensor_data, h5py.Group):
                            if "rgb" in sensor_data:
                                data["obs"][f"image_{sensor_name}"] = sensor_data["rgb"][:]
                                if verbose:
                                    print(f"    Found image for {sensor_name}: shape={sensor_data['rgb'].shape}")
                
                if verbose:
                    print(f"    Extracted observation keys: {list(data['obs'].keys())}")
            else:
                if verbose:
                    print(f"    Warning: No 'obs' found. Available keys: {list(traj_group.keys())}")
                data["obs"] = {}
            
            # Read info
            if "info" in traj_group:
                info_group = traj_group["info"]
                data["info"] = {}
                for key in info_group.keys():
                    data["info"][key] = info_group[key][:]
            
            # Read episode data
            if "episode_data" in traj_group:
                ep_group = traj_group["episode_data"]
                data["episode_data"] = {}
                for key in ep_group.keys():
                    val = ep_group[key]
                    if isinstance(val, h5py.Dataset):
                        data["episode_data"][key] = val[()]
                    else:
                        data["episode_data"][key] = val
        else:
            # No trajectory group, try direct access
            if verbose:
                print("    No 'trajectory' group found, trying direct access")
            
            # Try to find action directly
            if "action" in f:
                data["action"] = f["action"][:]
                if verbose:
                    print(f"    Found actions at top level: shape={data['action'].shape}")
            else:
                raise ValueError(f"No 'trajectory' group and no 'action' found. Top-level keys: {list(f.keys())}")
            
            # Try to find obs directly
            if "obs" in f:
                obs_data = f["obs"]
                data["obs"] = {}
                if isinstance(obs_data, h5py.Group):
                    for key in obs_data.keys():
                        data["obs"][key] = obs_data[key][:]
                else:
                    data["obs"]["obs"] = obs_data[:]
            else:
                data["obs"] = {}
        
        # Read metadata
        if "meta" in f:
            meta_group = f["meta"]
            data["meta"] = {}
            for key in meta_group.keys():
                val = meta_group[key]
                if isinstance(val, h5py.Dataset):
                    try:
                        # Try to read as string
                        data["meta"][key] = val.asstr()[()]
                    except:
                        data["meta"][key] = val[()]
                else:
                    data["meta"][key] = val
    finally:
        f.close()
    
    return data


def extract_robot_state(obs_dict: Dict[str, np.ndarray], step_idx: int) -> np.ndarray:
    """
    Extract robot joint positions from ManiSkill observation at a specific step.
    For dual-arm: returns concatenated [left_arm_qpos, right_arm_qpos] (12-dim)
    For single-arm: returns arm_qpos (6-dim)
    
    Expected structure: obs_dict["agent_qpos"] contains (T, state_dim) array
    """
    # New structure: agent_qpos is already extracted and concatenated
    if "agent_qpos" in obs_dict:
        qpos = obs_dict["agent_qpos"]
        if isinstance(qpos, np.ndarray):
            if qpos.ndim > 1 and step_idx < len(qpos):
                qpos_val = qpos[step_idx]
            else:
                qpos_val = qpos
            # qpos_val should already be arm joints only (6 or 12 dim)
            return qpos_val
    
    # Fallback: try old structure
    if "agent" in obs_dict:
        agent_data = obs_dict["agent"]
        if isinstance(agent_data, dict) and "qpos" in agent_data:
            qpos = agent_data["qpos"]
            if isinstance(qpos, np.ndarray) and step_idx < len(qpos):
                qpos_val = qpos[step_idx] if qpos.ndim > 1 else qpos
                if qpos_val.shape[-1] >= 6:
                    return qpos_val[..., :6]
                return qpos_val
    
    # Direct qpos key
    if "qpos" in obs_dict:
        qpos = obs_dict["qpos"]
        if isinstance(qpos, np.ndarray):
            if qpos.ndim > 1 and step_idx < len(qpos):
                qpos_val = qpos[step_idx]
            else:
                qpos_val = qpos
            if qpos_val.shape[-1] >= 6:
                return qpos_val[..., :6]
            return qpos_val
    
    raise ValueError(f"Could not extract robot state from observation at step {step_idx}. Available keys: {list(obs_dict.keys())}")


def extract_images(obs_dict: Dict[str, np.ndarray], step_idx: int, 
                   apply_distortion_to_front: bool = True) -> Dict[str, np.ndarray]:
    """
    Extract camera images from ManiSkill observation at a specific step.
    Returns dict with 'front', 'left_wrist', 'right_wrist' images.
    
    Expected structure: obs_dict["image_front"] contains (T, H, W, C) array
    """
    images = {}
    
    # New structure: image_front, image_left_wrist, etc.
    if "image_front" in obs_dict:
        front_img = obs_dict["image_front"]
        if isinstance(front_img, np.ndarray):
            if front_img.ndim == 4:  # (T, H, W, C)
                front_img = front_img[step_idx]
            elif front_img.ndim == 5:  # (B, T, H, W, C)
                front_img = front_img[0, step_idx]
            
            # Handle RGBA -> RGB
            if front_img.shape[-1] == 4:
                front_img = front_img[..., :3]
            
            # Ensure uint8 (should already be uint8 from ManiSkill)
            if front_img.dtype != np.uint8:
                if front_img.max() <= 1.0:
                    front_img = (front_img * 255).astype(np.uint8)
                else:
                    front_img = front_img.astype(np.uint8)
            
            # Resize if needed to match expected size (480x640)
            if front_img.shape[:2] != (480, 640):
                front_img = cv2.resize(front_img, (640, 480))
            
            # Apply distortion if requested
            if apply_distortion_to_front:
                front_img = apply_distortion(front_img)
            
            images["front"] = front_img
    
    # Try left_wrist and right_wrist
    for wrist_name in ["left_wrist", "right_wrist"]:
        key = f"image_{wrist_name}"
        if key in obs_dict:
            wrist_img = obs_dict[key]
            if isinstance(wrist_img, np.ndarray):
                if wrist_img.ndim == 4:
                    wrist_img = wrist_img[step_idx]
                elif wrist_img.ndim == 5:
                    wrist_img = wrist_img[0, step_idx]
                
                if wrist_img.shape[-1] == 4:
                    wrist_img = wrist_img[..., :3]
                
                if wrist_img.dtype != np.uint8:
                    if wrist_img.max() <= 1.0:
                        wrist_img = (wrist_img * 255).astype(np.uint8)
                    else:
                        wrist_img = wrist_img.astype(np.uint8)
                
                if wrist_img.shape[:2] != (480, 640):
                    wrist_img = cv2.resize(wrist_img, (640, 480))
                
                images[wrist_name] = wrist_img
    
    # Fallback: try old structure
    if "front" not in images:
        for img_key in ["image", "images", "rgb"]:
            if img_key in obs_dict:
                img_data = obs_dict[img_key]
                if isinstance(img_data, dict) and "front" in img_data:
                    front_img = img_data["front"]
                    if isinstance(front_img, np.ndarray):
                        if front_img.ndim == 4:
                            front_img = front_img[step_idx]
                        elif front_img.ndim == 5:
                            front_img = front_img[0, step_idx]
                        
                        if front_img.shape[-1] == 4:
                            front_img = front_img[..., :3]
                        
                        if front_img.dtype != np.uint8:
                            if front_img.max() <= 1.0:
                                front_img = (front_img * 255).astype(np.uint8)
                            else:
                                front_img = front_img.astype(np.uint8)
                        
                        if front_img.shape[:2] != (480, 640):
                            front_img = cv2.resize(front_img, (640, 480))
                        
                        if apply_distortion_to_front:
                            front_img = apply_distortion(front_img)
                        
                        images["front"] = front_img
                        break
    
    # Placeholder for missing wrist cameras
    if "left_wrist" not in images:
        images["left_wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
    if "right_wrist" not in images:
        images["right_wrist"] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    return images


def convert_single_trajectory(input_h5_path: pathlib.Path, output_dir: pathlib.Path,
                              task_name: str, apply_distortion_to_front: bool = True,
                              episode_index: int = 0, traj_index: int = 0, verbose: bool = False) -> bool:
    """
    Convert a single ManiSkill trajectory to LeRobot format.
    
    Returns True if successful, False otherwise.
    """
    try:
        # Read ManiSkill trajectory
        if verbose:
            print(f"  Reading {input_h5_path}...")
        
        traj_data = read_maniskill_trajectory(input_h5_path, verbose=verbose, traj_index=traj_index)
        
        if "action" not in traj_data or len(traj_data["action"]) == 0:
            print(f"  Error: No actions found in {input_h5_path}")
            print(f"  This trajectory file may have been recorded with obs_mode='none'.")
            print(f"  You need to replay the trajectory to generate observations and actions.")
            print(f"  Try using: mani_skill.trajectory.replay_trajectory")
            return False
        
        num_steps = len(traj_data["action"])
        # Note: observations might have T+1 steps (includes initial state)
        # Actions have T steps (one per transition)
        if verbose:
            print(f"  Found {num_steps} action steps")
            if "obs" in traj_data and "agent_qpos" in traj_data["obs"]:
                obs_steps = len(traj_data["obs"]["agent_qpos"])
                print(f"  Found {obs_steps} observation steps")
        
        # Get task prompt
        task_prompt_map = {
            "lift": "Pick up the red cube and lift it.",
            "stack": "Stack the red cube on top of the green cube.",
            "sort": "Move the red cube to the left region and the green cube to the right region.",
        }
        task_prompt = task_prompt_map.get(task_name, "")
        
        # Extract data for each step
        states = []
        front_images = []
        left_wrist_images = []
        right_wrist_images = []
        actions = []
        episode_indices = []
        task_prompts = []
        
        # Use progress bar for steps if there are many
        step_iter = range(num_steps)
        if num_steps > 100:
            step_iter = tqdm(step_iter, desc=f"    Processing {num_steps} steps", leave=False, unit="step")
        
        for step_idx in step_iter:
            # Extract state (observations have T+1 steps, actions have T steps)
            # Use step_idx for actions, step_idx+1 for next state if available
            obs_step_idx = min(step_idx, num_steps)  # Cap at available observations
            
            try:
                state = extract_robot_state(traj_data["obs"], obs_step_idx)
                states.append(state)
            except Exception as e:
                if verbose:
                    print(f"  Warning at step {step_idx}: Could not extract state: {e}")
                # Use previous state or zeros
                if len(states) > 0:
                    states.append(states[-1].copy())
                else:
                    # Default: try to infer dimension from action
                    if len(traj_data["action"]) > 0:
                        action_dim = traj_data["action"][0].shape[-1]
                        # For dual-arm, action might be 12-dim (6+6)
                        # State should match action dimension (arm joints only)
                        state_dim = min(12, action_dim)
                    else:
                        state_dim = 6
                    states.append(np.zeros(state_dim))
            
            # Extract images
            try:
                images = extract_images(traj_data["obs"], obs_step_idx, apply_distortion_to_front)
                front_images.append(images.get("front", np.zeros((480, 640, 3), dtype=np.uint8)))
                left_wrist_images.append(images.get("left_wrist", np.zeros((480, 640, 3), dtype=np.uint8)))
                right_wrist_images.append(images.get("right_wrist", np.zeros((480, 640, 3), dtype=np.uint8)))
            except Exception as e:
                if verbose:
                    print(f"  Warning at step {step_idx}: Could not extract images: {e}")
                front_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
                left_wrist_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
                right_wrist_images.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
            # Extract action
            action = traj_data["action"][step_idx]
            # Actions are already joint positions (including gripper potentially)
            # Extract arm joints only (first 6 or 12)
            if action.shape[-1] > 12:
                # Likely includes gripper, take arm joints only
                action = action[..., :12]
            elif action.shape[-1] > 6:
                # Dual-arm: 12-dim is fine
                action = action[..., :12]
            else:
                # Single-arm: 6-dim is fine
                action = action[..., :6]
            actions.append(action)
            
            episode_indices.append(episode_index)
            task_prompts.append(task_prompt)
        
        # Convert to numpy arrays
        states = np.array(states)
        front_images = np.array(front_images)
        left_wrist_images = np.array(left_wrist_images)
        right_wrist_images = np.array(right_wrist_images)
        actions = np.array(actions)
        episode_indices = np.array(episode_indices, dtype=np.int64)
        
        # Save as npz file (intermediate format)
        # LeRobot Dataset format will be created separately if needed
        output_file = output_dir / f"episode_{episode_index:05d}.npz"
        
        np.savez_compressed(
            output_file,
            observation_state=states,
            observation_images_front=front_images,
            observation_images_left_wrist=left_wrist_images,
            observation_images_right_wrist=right_wrist_images,
            action=actions,
            episode_index=episode_indices,
            task=np.array(task_prompts, dtype=object),
        )
        
        if verbose:
            print(f"  Saved to {output_file}")
            print(f"    States shape: {states.shape}")
            print(f"    Front images shape: {front_images.shape}")
            print(f"    Actions shape: {actions.shape}")
        
        return True
        
    except Exception as e:
        print(f"  Error converting {input_h5_path}: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return False


def main():
    args = parse_args()
    
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all h5 files
    h5_files = sorted(list(input_dir.glob("*.h5")))
    
    print(f"Found {len(h5_files)} trajectory files")
    print(f"Output directory: {output_dir}")
    print(f"Task: {args.task_name}")
    print(f"Apply distortion: {args.apply_distortion}")
    
    if len(h5_files) == 0:
        print("Error: No h5 files found in input directory!")
        return
    
    # Inspect first file if requested
    if args.inspect:
        print("\nInspecting first trajectory file structure:")
        inspect_h5_structure(h5_files[0])
        return
    
    # Convert each trajectory
    # Each h5 file may contain multiple trajectories (traj_0, traj_1, ...)
    successful = 0
    failed = 0
    episode_idx = 0
    
    for h5_file in tqdm(h5_files, desc="Converting trajectories"):
        # First, check how many trajectories are in this file
        try:
            with h5py.File(h5_file, 'r') as f:
                traj_keys = [key for key in f.keys() if key.startswith("traj")]
                num_trajs_in_file = len(traj_keys)
        except (RuntimeError, OSError, IOError) as e:
            print(f"  Error: Failed to read {h5_file}: {e}")
            print(f"    This file may be corrupted. Skipping...")
            failed += 1
            continue
        
        if num_trajs_in_file == 0:
            print(f"  Warning: No trajectories found in {h5_file}")
            failed += 1
            continue
        
        # Convert each trajectory in the file
        for traj_idx in range(num_trajs_in_file):
            try:
                print(f"  Converting trajectory {traj_idx+1}/{num_trajs_in_file} from {h5_file.name}...")
                if convert_single_trajectory(h5_file, output_dir, args.task_name,
                                           args.apply_distortion, episode_idx, traj_idx, args.verbose):
                    successful += 1
                    episode_idx += 1
                    print(f"    ✓ Successfully converted to episode_{episode_idx-1:05d}.npz")
                else:
                    failed += 1
                    episode_idx += 1  # Still increment to avoid overwriting
                    print(f"    ✗ Failed to convert trajectory {traj_idx}")
            except (RuntimeError, OSError, IOError, KeyError, ValueError) as e:
                print(f"    ✗ Error converting trajectory {traj_idx} in {h5_file}: {e}")
                failed += 1
                episode_idx += 1  # Still increment to avoid overwriting
            except KeyboardInterrupt:
                print(f"\n  Interrupted by user. Processed {successful} successful, {failed} failed so far.")
                raise
    
    print(f"\nConversion complete!")
    print(f"Successful: {successful}/{len(h5_files)}")
    print(f"Failed: {failed}/{len(h5_files)}")
    print(f"\nConverted trajectories saved to: {output_dir}")
    print(f"Format: .npz files with keys:")
    print(f"  - observation_state: robot joint positions")
    print(f"  - observation_images_front: front camera images")
    print(f"  - observation_images_left_wrist: left wrist camera images")
    print(f"  - observation_images_right_wrist: right wrist camera images")
    print(f"  - action: robot actions (joint positions)")
    print(f"  - episode_index: episode indices")
    print(f"  - task: task prompts")


if __name__ == "__main__":
    main()
