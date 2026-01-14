"""
自定义训练脚本：直接加载 .npz 文件训练 Diffusion Policy

这个脚本直接使用 LeRobot 的 DiffusionPolicy 模型类，但使用自定义的训练循环，
避免了 LeRobot 训练命令的复杂性。
"""

import argparse
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from typing import Dict, Any, Optional
import time

try:
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("Error: LeRobot not installed. Please install it: pip install lerobot")


class NPZDataset(Dataset):
    """Dataset class for loading .npz files."""
    
    def __init__(self, npz_files, use_wrist_cameras=True, horizon=16, n_obs_steps=1):
        self.npz_files = sorted(npz_files)
        self.use_wrist_cameras = use_wrist_cameras
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        print(f"Loaded {len(self.npz_files)} episodes")
    
    def __len__(self):
        return len(self.npz_files)
    
    def __getitem__(self, idx):
        data = np.load(self.npz_files[idx], allow_pickle=True)
        
        num_frames = len(data["observation_state"])
        
        # Sample a starting frame such that we can extract a sequence of length horizon
        # If episode is shorter than horizon, start from frame 0
        max_start_frame = max(0, num_frames - self.horizon)
        start_frame = np.random.randint(0, max_start_frame + 1) if max_start_frame > 0 else 0
        
        # Extract observation sequence (n_obs_steps frames)
        # For n_obs_steps=1, we just use the start frame
        # For n_obs_steps>1, we use frames [start_frame-n_obs_steps+1:start_frame+1]
        obs_start_frame = max(0, start_frame - self.n_obs_steps + 1)
        obs_end_frame = start_frame + 1
        
        # Extract state sequence: shape (n_obs_steps, state_dim)
        state_sequence = data["observation_state"][obs_start_frame:obs_end_frame].astype(np.float32)
        if len(state_sequence) < self.n_obs_steps:
            # Pad with first frame if needed
            padding = np.tile(state_sequence[0:1], (self.n_obs_steps - len(state_sequence), 1))
            state_sequence = np.concatenate([padding, state_sequence], axis=0)
        state_sequence = torch.from_numpy(state_sequence)
        
        # Extract image sequence: shape (n_obs_steps, H, W, C)
        front_image_sequence = data["observation_images_front"][obs_start_frame:obs_end_frame].astype(np.uint8)
        if len(front_image_sequence) < self.n_obs_steps:
            # Pad with first frame if needed
            padding = np.tile(front_image_sequence[0:1], (self.n_obs_steps - len(front_image_sequence), 1, 1, 1))
            front_image_sequence = np.concatenate([padding, front_image_sequence], axis=0)
        front_image_sequence = torch.from_numpy(front_image_sequence)
        
        # Extract action sequence (horizon steps)
        # Shape: (horizon, action_dim)
        end_frame = min(start_frame + self.horizon, num_frames)
        action_sequence = data["action"][start_frame:end_frame].astype(np.float32)
        
        # Pad if necessary (shouldn't happen often, but handle it)
        if len(action_sequence) < self.horizon:
            padding = np.tile(action_sequence[-1:], (self.horizon - len(action_sequence), 1))
            action_sequence = np.concatenate([action_sequence, padding], axis=0)
        
        action_sequence = torch.from_numpy(action_sequence)
        
        result = {
            "state": state_sequence,  # Shape: (n_obs_steps, state_dim)
            "front_image": front_image_sequence,  # Shape: (n_obs_steps, H, W, C)
            "action": action_sequence,  # Shape: (horizon, action_dim)
        }
        
        # Add wrist cameras if available
        if self.use_wrist_cameras and "observation_images_left_wrist" in data:
            left_wrist_sequence = data["observation_images_left_wrist"][obs_start_frame:obs_end_frame].astype(np.uint8)
            if len(left_wrist_sequence) < self.n_obs_steps:
                padding = np.tile(left_wrist_sequence[0:1], (self.n_obs_steps - len(left_wrist_sequence), 1, 1, 1))
                left_wrist_sequence = np.concatenate([padding, left_wrist_sequence], axis=0)
            result["left_wrist_image"] = torch.from_numpy(left_wrist_sequence)
        
        if self.use_wrist_cameras and "observation_images_right_wrist" in data:
            right_wrist_sequence = data["observation_images_right_wrist"][obs_start_frame:obs_end_frame].astype(np.uint8)
            if len(right_wrist_sequence) < self.n_obs_steps:
                padding = np.tile(right_wrist_sequence[0:1], (self.n_obs_steps - len(right_wrist_sequence), 1, 1, 1))
                right_wrist_sequence = np.concatenate([padding, right_wrist_sequence], axis=0)
            result["right_wrist_image"] = torch.from_numpy(right_wrist_sequence)
        
        return result


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences."""
    # Stack all items
    result = {}
    for key in batch[0].keys():
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([item[key] for item in batch])
        else:
            result[key] = [item[key] for item in batch]
    return result


def compute_dataset_stats(
    npz_files: list,
    use_wrist_cameras: bool = True,
    num_samples: int = 1000,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute dataset statistics (min, max, mean, std) for normalization.
    
    Args:
        npz_files: List of .npz file paths
        use_wrist_cameras: Whether to include wrist cameras
        num_samples: Number of random samples to use for statistics
    
    Returns:
        Dictionary with stats for each feature
    """
    print(f"Computing dataset statistics from {min(num_samples, len(npz_files) * 100)} samples...")
    
    all_states = []
    all_actions = []
    
    # Sample random frames from episodes
    np.random.seed(42)
    samples_collected = 0
    
    for npz_file in npz_files:
        if samples_collected >= num_samples:
            break
        
        data = np.load(npz_file, allow_pickle=True)
        num_frames = len(data["observation_state"])
        
        # Sample a few frames from this episode
        num_samples_from_episode = min(5, num_frames, num_samples - samples_collected)
        frame_indices = np.random.choice(num_frames, num_samples_from_episode, replace=False)
        
        for frame_idx in frame_indices:
            all_states.append(data["observation_state"][frame_idx])
            all_actions.append(data["action"][frame_idx])
            samples_collected += 1
            
            if samples_collected >= num_samples:
                break
    
    # Convert to numpy arrays
    all_states = np.array(all_states, dtype=np.float32)
    all_actions = np.array(all_actions, dtype=np.float32)
    
    # ImageNet statistics for RGB images (used for MEAN_STD normalization)
    # ImageNet mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225]
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    # Compute statistics
    stats = {
        "observation.state": {
            "min": torch.from_numpy(all_states.min(axis=0)),
            "max": torch.from_numpy(all_states.max(axis=0)),
            "mean": torch.from_numpy(all_states.mean(axis=0)),
            "std": torch.from_numpy(all_states.std(axis=0)) + 1e-6,  # Add small epsilon to avoid division by zero
        },
        "action": {
            "min": torch.from_numpy(all_actions.min(axis=0)),
            "max": torch.from_numpy(all_actions.max(axis=0)),
            "mean": torch.from_numpy(all_actions.mean(axis=0)),
            "std": torch.from_numpy(all_actions.std(axis=0)) + 1e-6,
        },
        # Image statistics (ImageNet values for MEAN_STD normalization)
        "observation.images.front": {
            "mean": imagenet_mean,
            "std": imagenet_std,
        },
    }
    
    # Add wrist camera stats if needed
    if use_wrist_cameras:
        stats["observation.images.left_wrist"] = {
            "mean": imagenet_mean,
            "std": imagenet_std,
        }
        stats["observation.images.right_wrist"] = {
            "mean": imagenet_mean,
            "std": imagenet_std,
        }
    
    print(f"Computed statistics from {samples_collected} samples")
    print(f"  State: min={stats['observation.state']['min'].min():.3f}, max={stats['observation.state']['max'].max():.3f}")
    print(f"  Action: min={stats['action']['min'].min():.3f}, max={stats['action']['max'].max():.3f}")
    print(f"  Images: Using ImageNet statistics (MEAN_STD normalization)")
    
    return stats


def create_model(
    config_dict: Dict[str, Any],
    device: torch.device,
    dataset_stats: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> DiffusionPolicy:
    """Create Diffusion Policy model from config dictionary."""
    if not LEROBOT_AVAILABLE:
        raise ImportError("LeRobot not installed")
    
    # Create config object
    policy_config = DiffusionConfig(**config_dict)
    
    # Create model with dataset stats
    model = DiffusionPolicy(policy_config, dataset_stats=dataset_stats)
    model.to(device)
    
    return model


def train_step(
    model: DiffusionPolicy,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    use_wrist_cameras: bool = True,
):
    """Single training step."""
    # Move to device
    # States: (B, n_obs_steps, state_dim)
    states = batch["state"].to(device)
    # Images: (B, n_obs_steps, H, W, C)
    front_images = batch["front_image"].to(device)
    # Actions: (B, horizon, action_dim)
    actions = batch["action"].to(device)
    
    # Convert images from (B, n_obs_steps, H, W, C) to (B, n_obs_steps, C, H, W) and normalize to [0, 1]
    # LeRobot expects: (B, n_obs_steps, num_cameras, C, H, W) or (B, n_obs_steps, C, H, W) for single camera
    front_images = front_images.permute(0, 1, 4, 2, 3).float() / 255.0  # (B, n_obs_steps, C, H, W)
    
    batch_size = states.shape[0]
    
    # Actions should now be (B, horizon, action_dim)
    # action_is_pad should be (B, horizon) - marks which timesteps are padding
    # Since we extract full sequences, all are real (not padding)
    action_is_pad = torch.zeros(batch_size, actions.shape[1], dtype=torch.bool, device=device)
    
    # Prepare batch dict (LeRobot format)
    # DiffusionPolicy.forward expects:
    # - observation.state: (B, n_obs_steps, state_dim)
    # - observation.images.front: (B, n_obs_steps, C, H, W) or dict with camera names
    # - action: (B, horizon, action_dim)
    model_batch = {
        "observation.state": states,  # (B, n_obs_steps, state_dim)
        "observation.images.front": front_images,  # (B, n_obs_steps, C, H, W)
        "action": actions,  # (B, horizon, action_dim)
        "action_is_pad": action_is_pad,  # (B, horizon)
    }
    
    # Add wrist cameras if available
    if use_wrist_cameras and "left_wrist_image" in batch:
        left_wrist = batch["left_wrist_image"].to(device)  # (B, n_obs_steps, H, W, C)
        left_wrist = left_wrist.permute(0, 1, 4, 2, 3).float() / 255.0  # (B, n_obs_steps, C, H, W)
        model_batch["observation.images.left_wrist"] = left_wrist
    
    if use_wrist_cameras and "right_wrist_image" in batch:
        right_wrist = batch["right_wrist_image"].to(device)  # (B, n_obs_steps, H, W, C)
        right_wrist = right_wrist.permute(0, 1, 4, 2, 3).float() / 255.0  # (B, n_obs_steps, C, H, W)
        model_batch["observation.images.right_wrist"] = right_wrist
    
    # Forward pass - compute loss
    # DiffusionPolicy.forward returns (loss, None)
    loss, _ = model(model_batch)
    
    return loss


def train(
    dataset_dir: str,
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    save_freq: int = 10,
    num_workers: int = 4,
    use_wrist_cameras: bool = True,
    config_overrides: Optional[Dict[str, Any]] = None,
):
    """Main training function."""
    if not LEROBOT_AVAILABLE:
        raise ImportError("LeRobot not installed. Please install it: pip install lerobot")
    
    dataset_path = pathlib.Path(dataset_dir)
    output_path = pathlib.Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device)
    
    # Find all .npz files
    npz_files = list(dataset_path.glob("episode_*.npz"))
    if len(npz_files) == 0:
        raise ValueError(f"No .npz files found in {dataset_path}")
    
    print(f"Found {len(npz_files)} episodes in {dataset_path}")
    
    # Load first file to get dimensions
    sample_data = np.load(npz_files[0], allow_pickle=True)
    state_dim = sample_data["observation_state"].shape[-1]
    action_dim = sample_data["action"].shape[-1]
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Check if wrist cameras are available
    has_left_wrist = "observation_images_left_wrist" in sample_data
    has_right_wrist = "observation_images_right_wrist" in sample_data
    if use_wrist_cameras:
        print(f"Wrist cameras: left={has_left_wrist}, right={has_right_wrist}")
    
    # Build input_features dict
    # PolicyFeature requires type and shape
    # Note: LeRobot expects channel-first format (C, H, W) for images
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(state_dim,)),
        "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    }
    
    # Add wrist cameras if available and requested
    if use_wrist_cameras:
        if has_left_wrist:
            input_features["observation.images.left_wrist"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
        if has_right_wrist:
            input_features["observation.images.right_wrist"] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
    
    # Build output_features dict
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(action_dim,)),
    }
    
    # Set normalization mapping
    # Images use MEAN_STD (ImageNet statistics), state and action use MIN_MAX (from dataset)
    normalization_mapping = {
        "observation.state": NormalizationMode.MIN_MAX,
        "action": NormalizationMode.MIN_MAX,
    }
    # Add image features - use MEAN_STD (ImageNet default)
    normalization_mapping["observation.images.front"] = NormalizationMode.MEAN_STD
    if use_wrist_cameras:
        if has_left_wrist:
            normalization_mapping["observation.images.left_wrist"] = NormalizationMode.MEAN_STD
        if has_right_wrist:
            normalization_mapping["observation.images.right_wrist"] = NormalizationMode.MEAN_STD
    
    # Default model config
    # Note: input_features and output_features specify the observation and action structure
    # LeRobot will infer dimensions from these features
    # Important: If using pretrained weights, use_group_norm must be False (pretrained models use BatchNorm)
    model_config = {
        "n_obs_steps": 1,
        "horizon": 16,
        "n_action_steps": 8,
        "vision_backbone": "resnet18",
        "pretrained_backbone_weights": "IMAGENET1K_V1",
        "crop_shape": None,  # Use full image (480x640)
        "crop_is_random": True,
        "use_group_norm": False,  # Must be False when using pretrained weights (they use BatchNorm)
        "down_dims": [512, 1024, 2048],
        "kernel_size": 5,
        "n_groups": 8,
        "diffusion_step_embed_dim": 128,
        "use_film_scale_modulation": True,
        "noise_scheduler_type": "DDPM",
        "num_train_timesteps": 100,
        "beta_schedule": "squaredcos_cap_v2",
        "beta_start": 0.0001,
        "beta_end": 0.02,
        "prediction_type": "epsilon",
        "clip_sample": True,
        "clip_sample_range": 1.0,
        "num_inference_steps": 16,
        "input_features": input_features,
        "output_features": output_features,
        "normalization_mapping": normalization_mapping,
    }
    
    # Apply config overrides
    if config_overrides:
        model_config.update(config_overrides)
    
    # Get horizon and n_obs_steps from config (needed for dataset)
    horizon = model_config.get("horizon", 16)
    n_obs_steps = model_config.get("n_obs_steps", 1)
    
    # Create dataset and dataloader (need horizon and n_obs_steps)
    dataset = NPZDataset(npz_files, use_wrist_cameras=use_wrist_cameras, horizon=horizon, n_obs_steps=n_obs_steps)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Compute dataset statistics for normalization
    print("Computing dataset statistics...")
    dataset_stats = compute_dataset_stats(npz_files, use_wrist_cameras=use_wrist_cameras)
    
    print("Creating model...")
    model = create_model(model_config, device, dataset_stats=dataset_stats)
    model.train()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=1e-6,
    )
    
    print(f"\nStarting training:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Save frequency: every {save_freq} epochs")
    print()
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_start_time = time.time()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()
            
            loss = train_step(model, batch, device, use_wrist_cameras=use_wrist_cameras)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            loss_val = loss.item()
            epoch_losses.append(loss_val)
            global_step += 1
            
            pbar.set_postfix({
                "loss": f"{loss_val:.6f}",
                "avg_loss": f"{np.mean(epoch_losses):.6f}",
            })
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_losses)
        epoch_time = time.time() - epoch_start_time
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f} - LR: {current_lr:.2e} - Time: {epoch_time:.1f}s")
        
        # Save checkpoint
        if (epoch + 1) % save_freq == 0 or epoch == num_epochs - 1:
            checkpoint_dir = output_path / f"checkpoint-epoch-{epoch+1}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model (LeRobot format)
            model.save_pretrained(str(checkpoint_dir))
            
            # Save training info
            # Convert model_config to JSON-serializable format
            serializable_config = {}
            for key, value in model_config.items():
                if isinstance(value, (PolicyFeature,)):
                    # Convert PolicyFeature to dict
                    serializable_config[key] = {
                        "type": value.type.name if hasattr(value.type, 'name') else str(value.type),
                        "shape": list(value.shape),
                    }
                elif isinstance(value, (NormalizationMode,)):
                    # Convert NormalizationMode to string
                    serializable_config[key] = value.name if hasattr(value, 'name') else str(value)
                elif isinstance(value, dict):
                    # Recursively handle dicts
                    serializable_dict = {}
                    for k, v in value.items():
                        if isinstance(v, (PolicyFeature,)):
                            serializable_dict[k] = {
                                "type": v.type.name if hasattr(v.type, 'name') else str(v.type),
                                "shape": list(v.shape),
                            }
                        elif isinstance(v, (NormalizationMode,)):
                            serializable_dict[k] = v.name if hasattr(v, 'name') else str(v)
                        else:
                            serializable_dict[k] = v
                    serializable_config[key] = serializable_dict
                else:
                    serializable_config[key] = value
            
            training_info = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "learning_rate": current_lr,
                "global_step": global_step,
                "model_config": serializable_config,
            }
            with open(checkpoint_dir / "training_info.json", "w") as f:
                json.dump(training_info, f, indent=2)
            
            print(f"  Saved checkpoint to {checkpoint_dir}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_dir = output_path / "checkpoint-best"
            best_checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_checkpoint_dir))
            print(f"  Saved best model (loss: {best_loss:.6f}) to {best_checkpoint_dir}")
    
    print("\nTraining completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"All checkpoints saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy (custom training loop)")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory containing .npz files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=10,
        help="Save checkpoint every N epochs"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--no-wrist-cameras",
        action="store_true",
        help="Don't use wrist cameras (only front camera)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    train(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        save_freq=args.save_freq,
        num_workers=args.num_workers,
        use_wrist_cameras=not args.no_wrist_cameras,
    )


if __name__ == "__main__":
    main()
