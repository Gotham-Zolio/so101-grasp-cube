"""
Convert ManiSkill trajectory (h5 format) to LeRobot Dataset format (parquet).

This script reads ManiSkill trajectory files and converts them to LeRobot Dataset format,
which uses parquet files for data storage. This format is compatible with LeRobot's
native training commands.
"""

import argparse
import h5py
import numpy as np
import pathlib
from tqdm import tqdm
import cv2
import json
from typing import Dict, Any, List, Optional
import torch

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Error: LeRobot not installed. Please install it: pip install lerobot")
    exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ManiSkill trajectories to LeRobot Dataset (parquet)")
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
        help="Task name (e.g., lift, stack, sort, avoid_obstacle)"
    )
    parser.add_argument(
        "--use-videos",
        action="store_true",
        default=True,
        help="Store images as MP4 videos (default: True, faster for training but slower conversion)"
    )
    parser.add_argument(
        "--no-videos",
        action="store_false",
        dest="use_videos",
        help="Disable video encoding (faster conversion but larger storage and slower training)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed conversion information"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Repository ID for the dataset (optional, defaults to task-name)"
    )
    return parser.parse_args()


def read_maniskill_trajectory(h5_path: pathlib.Path, verbose: bool = False, traj_index: int = 0) -> Dict[str, Any]:
    """
    Read ManiSkill trajectory h5 file.
    Returns dict with observations, actions, and metadata.
    """
    data = {}
    
    try:
        f = h5py.File(h5_path, 'r')
    except (RuntimeError, OSError, IOError) as e:
        raise RuntimeError(f"Failed to open h5 file {h5_path}: {e}") from e
    
    try:
        # Find trajectory group
        traj_keys = [k for k in f.keys() if k.startswith('traj')]
        if len(traj_keys) == 0:
            raise ValueError(f"No trajectory found in {h5_path}")
        
        traj_key = traj_keys[traj_index] if traj_index < len(traj_keys) else traj_keys[0]
        traj_group = f[traj_key]
        
        # Read actions
        if "actions" in traj_group:
            data["action"] = traj_group["actions"][:]
        elif "action" in traj_group:
            data["action"] = traj_group["action"][:]
        else:
            raise ValueError(f"No actions found in {h5_path}")
        
        # Read observations
        data["obs"] = {}
        if "obs" in traj_group:
            obs_group = traj_group["obs"]
            
            # Extract robot state (qpos)
            agent_qpos_list = []
            if "agent" in obs_group:
                agent_group = obs_group["agent"]
                for agent_key in agent_group.keys():
                    agent_data = agent_group[agent_key]
                    # Check if agent_data is a Group (has keys) and contains "qpos"
                    if isinstance(agent_data, h5py.Group) and "qpos" in agent_data:
                        qpos = agent_data["qpos"][:]
                        agent_qpos_list.append(qpos)
            
            if agent_qpos_list:
                # Concatenate all agent qpos (for dual-arm: left + right)
                # If multiple agents, concatenate along last dimension
                if len(agent_qpos_list) == 1:
                    data["obs"]["agent_qpos"] = agent_qpos_list[0]
                else:
                    # Stack along last dimension: (T, 6) + (T, 6) -> (T, 12)
                    data["obs"]["agent_qpos"] = np.concatenate(agent_qpos_list, axis=-1)
            
            # Extract sensor data (images)
            if "sensor_data" in obs_group:
                sensor_group = obs_group["sensor_data"]
                for sensor_name in sensor_group.keys():
                    sensor_data = sensor_group[sensor_name]
                    if isinstance(sensor_data, h5py.Group):
                        if "rgb" in sensor_data:
                            data["obs"][f"image_{sensor_name}"] = sensor_data["rgb"][:]
            
    
    finally:
        f.close()
    
    return data


def extract_robot_state(obs_dict: Dict[str, np.ndarray], step_idx: int) -> np.ndarray:
    """Extract robot joint positions from ManiSkill observation at a specific step."""
    if "agent_qpos" in obs_dict:
        qpos = obs_dict["agent_qpos"]
        if step_idx < len(qpos):
            return qpos[step_idx].astype(np.float32)
    
    raise ValueError(f"Could not extract robot state from observation at step {step_idx}")


def extract_images(obs_dict: Dict[str, np.ndarray], step_idx: int) -> Dict[str, np.ndarray]:
    """
    Extract camera images from ManiSkill observation at a specific step.
    Returns dict with available camera images (front, left_side, right_side).
    
    All tasks (lift, stack, sort, avoid_obstacle) use the same three cameras: front + left_side + right_side.
    
    Args:
        obs_dict: Observation dictionary from ManiSkill trajectory
        step_idx: Step index to extract
    """
    images = {}
    
    # Extract front camera
    if "image_front" in obs_dict:
        front_img = obs_dict["image_front"]
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
            
            images["front"] = front_img
    
    # Extract left_side and right_side cameras (fixed pose cameras for all tasks)
    for side_name in ["left_side", "right_side"]:
        key = f"image_{side_name}"
        if key in obs_dict:
            side_img = obs_dict[key]
            if isinstance(side_img, np.ndarray):
                if side_img.ndim == 4:
                    side_img = side_img[step_idx]
                elif side_img.ndim == 5:
                    side_img = side_img[0, step_idx]
                
                if side_img.shape[-1] == 4:
                    side_img = side_img[..., :3]
                
                if side_img.dtype != np.uint8:
                    if side_img.max() <= 1.0:
                        side_img = (side_img * 255).astype(np.uint8)
                    else:
                        side_img = side_img.astype(np.uint8)
                
                if side_img.shape[:2] != (480, 640):
                    side_img = cv2.resize(side_img, (640, 480))
                
                images[side_name] = side_img
    
    # Do NOT create placeholders - only return cameras that actually exist
    return images


def convert_h5_to_lerobot_dataset(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    task_name: str,
    repo_id: Optional[str] = None,
    use_videos: bool = True,
    verbose: bool = False,
):
    """Convert ManiSkill h5 files to LeRobot Dataset format."""
    
    if not LEROBOT_AVAILABLE:
        raise ImportError("LeRobot not installed")
    
    # Find all h5 files
    h5_files = sorted(list(input_dir.glob("*.h5")))
    if len(h5_files) == 0:
        raise ValueError(f"No .h5 files found in {input_dir}")
    
    print(f"Found {len(h5_files)} trajectory files")
    
    # Get task prompt
    task_prompt_map = {
        "lift": "Pick up the red cube and lift it.",
        "stack": "Stack the red cube on top of the green cube.",
        "sort": "Move the red cube to the left region and the green cube to the right region.",
        "avoid_obstacle": "Lift the red cube over obstacles to the middle. Lift the green cube over obstacles to the middle.",
    }
    task_prompt = task_prompt_map.get(task_name, "")
    
    # Determine dimensions from first file
    sample_data = read_maniskill_trajectory(h5_files[0], verbose=verbose)
    state_dim = sample_data["obs"]["agent_qpos"].shape[-1] if "agent_qpos" in sample_data["obs"] else 6
    action_dim = sample_data["action"].shape[-1]
    
    # Determine features (cameras and state/action)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": None,
        },
    }
    
    # All tasks use the same three cameras: front + left_side + right_side
    # Add image features
    # LeRobot uses "video" dtype when use_videos=True (encoded as MP4), "image" dtype when use_videos=False
    # Note: names must be ["height", "width", "channels"] for LeRobot to correctly parse dimensions
    image_dtype = "video" if use_videos else "image"
    
    if "image_front" in sample_data["obs"]:
        features["observation.images.front"] = {
            "dtype": image_dtype,
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    
    if "image_left_side" in sample_data["obs"]:
        features["observation.images.left_side"] = {
            "dtype": image_dtype,
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    
    if "image_right_side" in sample_data["obs"]:
        features["observation.images.right_side"] = {
            "dtype": image_dtype,
            "shape": (480, 640, 3),
            "names": ["height", "width", "channels"],
        }
    
    # Create LeRobot dataset
    repo_id = repo_id or f"{task_name}_cube"
    print(f"Creating LeRobot dataset: {repo_id}")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {action_dim}")
    print(f"  Features: {list(features.keys())}")
    print(f"  Image storage: {'videos (MP4)' if use_videos else 'images (arrays)'}")
    
    # LeRobotDataset.create() creates dataset at root/repo_id
    # But if root already exists, we need to use root/repo_id as the actual root
    # Actually, looking at the code, LeRobotDataset.create() uses root as the dataset root
    # So we should pass root/repo_id as root, or use repo_id and let it create in default location
    # Let's use a subdirectory approach: create dataset at output_dir/repo_id
    dataset_root = output_dir / repo_id
    
    # Check if dataset already exists
    if dataset_root.exists() and (dataset_root / "meta" / "info.json").exists():
        print(f"  Warning: Dataset directory {dataset_root} already exists.")
        print(f"  Loading existing dataset...")
        try:
            # Load existing dataset - pass dataset_root as root
            dataset = LeRobotDataset(repo_id=repo_id, root=dataset_root)
            print(f"  Existing dataset has {dataset.meta.total_episodes} episodes")
            print(f"  Will append new episodes to existing dataset...")
            # Get the next episode index
            episode_index = dataset.meta.total_episodes
        except Exception as e:
            raise FileExistsError(
                f"Directory {dataset_root} exists but is not a valid LeRobot dataset. "
                f"Please delete it or use a different repo_id. Error: {e}"
            ) from e
    else:
        # Create new dataset - use repo_id and set root to output_dir/repo_id
        # Actually, LeRobotDataset.create() expects root to be the parent directory
        # and creates root/repo_id. But we want output_dir/repo_id.
        # So we pass output_dir as root, and repo_id as repo_id
        # But that would create output_dir/repo_id, which is what we want!
        # However, if output_dir already exists, it will fail.
        # Let's create the dataset with root pointing to the parent of dataset_root
        if dataset_root.exists():
            # Remove existing directory if it's not a valid dataset
            import shutil
            print(f"  Removing existing directory {dataset_root}...")
            shutil.rmtree(dataset_root)
        
        # Create dataset - pass dataset_root as root
        # LeRobotDataset.create() will create the directory at root
        # Note: fps should be int, not float (LeRobot expects int)
        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=dataset_root,  # Dataset will be created at this path
            fps=30,  # ManiSkill default - use int
            features=features,
            use_videos=use_videos,  # Store images as videos (can be disabled for faster conversion)
        )
        episode_index = 0
    
    # Process each h5 file
    # Each h5 file may contain multiple trajectories (episodes)
    # episode_index is set above (0 for new dataset, or total_episodes for existing)
    import time
    start_time = time.time()
    
    for h5_file in tqdm(h5_files, desc="Converting h5 files"):
        try:
            # Check how many trajectories are in this h5 file
            with h5py.File(h5_file, 'r') as f:
                traj_keys = [k for k in f.keys() if k.startswith('traj')]
                num_trajectories = len(traj_keys)
            
            if num_trajectories == 0:
                print(f"  Warning: Skipping {h5_file.name} (no trajectories found)")
                continue
            
            if verbose:
                print(f"  Processing {h5_file.name}: {num_trajectories} trajectories")
            
            # Process each trajectory in the h5 file
            for traj_index in range(num_trajectories):
                try:
                    traj_data = read_maniskill_trajectory(h5_file, verbose=verbose, traj_index=traj_index)
                    
                    if "action" not in traj_data or len(traj_data["action"]) == 0:
                        if verbose:
                            print(f"    Warning: Skipping trajectory {traj_index} in {h5_file.name} (no actions)")
                        continue
                    
                    num_steps = len(traj_data["action"])
                    
                    # Add task to dataset if not exists
                    if task_prompt:
                        task_index = dataset.meta.get_task_index(task_prompt)
                        if task_index is None:
                            dataset.meta.add_task(task_prompt)
                    
                    # Add frames to episode buffer
                    for step_idx in range(num_steps):
                        # Extract state
                        try:
                            state = extract_robot_state(traj_data["obs"], step_idx)
                        except Exception as e:
                            if verbose:
                                print(f"      Warning at step {step_idx}: {e}")
                            state = np.zeros(state_dim, dtype=np.float32)
                        
                        # Extract images
                        # All tasks use the same three cameras: front + left_side + right_side
                        images = extract_images(traj_data["obs"], step_idx)
                        
                        # Extract action
                        action = traj_data["action"][step_idx].astype(np.float32)
                        if action.shape[-1] > action_dim:
                            action = action[..., :action_dim]
                        
                        # Create frame dict
                        frame = {
                            "observation.state": state,
                            "action": action,
                        }
                        
                        # Add images (all tasks use front + left_side + right_side)
                        if "front" in images:
                            frame["observation.images.front"] = images["front"]
                        if "left_side" in images:
                            frame["observation.images.left_side"] = images["left_side"]
                        if "right_side" in images:
                            frame["observation.images.right_side"] = images["right_side"]
                        
                        # Add frame to episode buffer
                        # Timestamp should be in seconds (float)
                        # Calculate timestamp as step_idx / fps
                        timestamp = float(step_idx) / 30.0  # fps = 30
                        # Ensure timestamp is a Python float, not numpy float
                        timestamp = float(timestamp)
                        dataset.add_frame(frame, task=task_prompt, timestamp=timestamp)
                    
                    # Save episode (this will write to disk and encode videos if use_videos=True)
                    episode_start_time = time.time()
                    dataset.save_episode()
                    episode_time = time.time() - episode_start_time
                    
                    episode_index += 1
                    
                    if verbose or (episode_index % 10 == 0):
                        elapsed_total = time.time() - start_time
                        avg_time_per_episode = elapsed_total / episode_index if episode_index > 0 else 0
                        print(f"    Episode {episode_index}: {num_steps} steps, {episode_time:.2f}s/ep, "
                              f"avg {avg_time_per_episode:.2f}s/ep")
                    
                except Exception as e:
                    print(f"    Error processing trajectory {traj_index} in {h5_file.name}: {e}")
                    if verbose:
                        import traceback
                        traceback.print_exc()
                    continue
            
        except Exception as e:
            print(f"  Error processing {h5_file.name}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue
    
    total_time = time.time() - start_time
    print(f"\nConversion complete!")
    print(f"  Total episodes: {episode_index}")
    print(f"  Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
    if episode_index > 0:
        print(f"  Average time per episode: {total_time/episode_index:.2f} seconds")
    dataset_root = output_dir / repo_id
    print(f"  Dataset saved to: {dataset_root}")
    print(f"\nYou can now train using:")
    print(f"  uv run python -m lerobot.scripts.train \\")
    print(f"    --dataset.root {dataset_root} \\")
    print(f"    --policy.type act")


def main():
    args = parse_args()
    
    input_dir = pathlib.Path(args.input_dir)
    output_dir = pathlib.Path(args.output_dir)
    # Don't create output_dir here - LeRobotDataset.create() will handle it
    # We'll pass output_dir/repo_id as root to LeRobotDataset.create()
    # So we only need to ensure the parent of output_dir exists
    if output_dir.parent != pathlib.Path('.'):
        output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    convert_h5_to_lerobot_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        task_name=args.task_name,
        repo_id=args.repo_id,
        use_videos=args.use_videos,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
