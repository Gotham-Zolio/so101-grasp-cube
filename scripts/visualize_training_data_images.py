"""
Visualize training data images from HDF5 files.

This script loads images from ManiSkill HDF5 trajectory files and displays them
to verify that training data contains valid (non-black) images.
"""

import argparse
import pathlib
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

def load_images_from_hdf5(hdf5_path: pathlib.Path, traj_idx: int = 0, max_images: int = 50, camera_name: str = None):
    """
    Load images from HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        traj_idx: Trajectory index
        max_images: Maximum number of images to load
        camera_name: Specific camera to load (None = all cameras, "front", "left_wrist", "right_wrist")
    
    Returns:
        dict: Dictionary mapping camera names to lists of images
    """
    all_images = {}
    
    with h5py.File(hdf5_path, 'r') as f:
        traj_keys = [k for k in f.keys() if k.startswith('traj_')]
        
        if len(traj_keys) == 0:
            print("No trajectories found in file")
            return all_images
        
        if traj_idx >= len(traj_keys):
            print(f"Trajectory index {traj_idx} out of range (max: {len(traj_keys)-1})")
            traj_idx = 0
        
        traj_key = traj_keys[traj_idx]
        traj_group = f[traj_key]
        
        print(f"Loading images from {traj_key}...")
        
        if "obs" not in traj_group:
            print("No 'obs' group found")
            return all_images
        
        obs_group = traj_group["obs"]
        
        if "sensor_data" not in obs_group:
            print("No 'sensor_data' found in obs group")
            return all_images
        
        sensor_group = obs_group["sensor_data"]
        
        # Determine which cameras to load
        camera_names = []
        if camera_name is not None:
            camera_names = [camera_name]
        else:
            # Load all available cameras
            available_sensors = list(sensor_group.keys())
            camera_names = available_sensors.copy()
            
            # For dual-arm tasks, try to map wrist_camera sensors to left_wrist/right_wrist
            # Check if we have wrist_camera sensors with agent prefixes (e.g., "so101-0_wrist_camera", "wrist_camera")
            wrist_cameras = [s for s in available_sensors if 'wrist' in s.lower()]
            
            # If we have wrist_camera sensors but they're not named left_wrist/right_wrist,
            # try to map them based on agent index or name pattern
            if wrist_cameras and 'left_wrist' not in available_sensors and 'right_wrist' not in available_sensors:
                print(f"Found wrist cameras with different names: {wrist_cameras}")
                print("  Attempting to map to left_wrist/right_wrist based on agent index...")
                
                # Sort wrist cameras by name to get consistent ordering
                # Names with "0" or "left" go to left_wrist, names with "1" or "right" go to right_wrist
                wrist_cameras_sorted = sorted(wrist_cameras)
                left_wrist_candidates = [s for s in wrist_cameras_sorted if '0' in s or 'left' in s.lower()]
                right_wrist_candidates = [s for s in wrist_cameras_sorted if '1' in s or 'right' in s.lower()]
                
                # If no clear pattern, use first for left, second for right
                if not left_wrist_candidates and not right_wrist_candidates:
                    if len(wrist_cameras_sorted) >= 2:
                        left_wrist_candidates = [wrist_cameras_sorted[0]]
                        right_wrist_candidates = [wrist_cameras_sorted[1]]
                    elif len(wrist_cameras_sorted) == 1:
                        # Single wrist camera -> right_wrist (for single-arm tasks)
                        right_wrist_candidates = [wrist_cameras_sorted[0]]
                
                print(f"  Mapping left_wrist candidates: {left_wrist_candidates}")
                print(f"  Mapping right_wrist candidates: {right_wrist_candidates}")
        
        print(f"Available sensors: {list(sensor_group.keys())}")
        print(f"Loading cameras: {camera_names}")
        
        # Build mapping from original sensor names to display names
        # For wrist cameras with agent prefixes, map to standard names
        sensor_name_mapping = {}
        wrist_cameras = [s for s in camera_names if 'wrist' in s.lower()]
        if wrist_cameras and 'left_wrist' not in camera_names and 'right_wrist' not in camera_names:
            wrist_cameras_sorted = sorted(wrist_cameras)
            left_wrist_candidates = [s for s in wrist_cameras_sorted if '0' in s or 'left' in s.lower()]
            right_wrist_candidates = [s for s in wrist_cameras_sorted if '1' in s or 'right' in s.lower()]
            
            # If no clear pattern, use first for left, second for right
            if not left_wrist_candidates and not right_wrist_candidates:
                if len(wrist_cameras_sorted) >= 2:
                    sensor_name_mapping[wrist_cameras_sorted[0]] = 'left_wrist'
                    sensor_name_mapping[wrist_cameras_sorted[1]] = 'right_wrist'
                elif len(wrist_cameras_sorted) == 1:
                    sensor_name_mapping[wrist_cameras_sorted[0]] = 'right_wrist'
            else:
                if left_wrist_candidates:
                    sensor_name_mapping[left_wrist_candidates[0]] = 'left_wrist'
                if right_wrist_candidates:
                    sensor_name_mapping[right_wrist_candidates[0]] = 'right_wrist'
                # Handle remaining wrist cameras
                remaining = [s for s in wrist_cameras_sorted if s not in sensor_name_mapping]
                if remaining:
                    if 'left_wrist' not in sensor_name_mapping.values():
                        sensor_name_mapping[remaining[0]] = 'left_wrist'
                    elif len(remaining) > 0:
                        sensor_name_mapping[remaining[0]] = 'right_wrist'
        
        for cam_name in camera_names:
            if cam_name not in sensor_group:
                print(f"  Camera '{cam_name}' not found, skipping")
                continue
            
            cam_sensor = sensor_group[cam_name]
            
            if "rgb" not in cam_sensor:
                print(f"  No 'rgb' data found in {cam_name} sensor")
                continue
            
            rgb_data = cam_sensor["rgb"]
            print(f"  Found {cam_name} RGB data: shape={rgb_data.shape}, dtype={rgb_data.dtype}")
            
            images = []
            num_images = min(max_images, rgb_data.shape[0])
            
            for i in range(num_images):
                img = rgb_data[i]
                
                # Handle different shapes
                if img.ndim == 3:
                    # Could be (H, W, C) or (C, H, W)
                    if img.shape[0] == 3 or img.shape[0] == 4:
                        # (C, H, W) -> (H, W, C)
                        img = img.transpose(1, 2, 0)
                    
                    # Convert to uint8 if needed
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = np.clip(img, 0, 255).astype(np.uint8)
                    
                    # Convert RGBA to RGB if needed
                    if img.shape[-1] == 4:
                        img = img[..., :3]
                    
                    # Ensure correct shape
                    if img.ndim == 3 and img.shape[-1] == 3:
                        images.append(img)
            
            if len(images) > 0:
                # Use mapped name if available, otherwise use original name
                display_name = sensor_name_mapping.get(cam_name, cam_name)
                all_images[display_name] = images
                if display_name != cam_name:
                    print(f"  Loaded {len(images)} images from {cam_name} camera (displayed as {display_name})")
                else:
                    print(f"  Loaded {len(images)} images from {cam_name} camera")
        
        print(f"Total cameras loaded: {len(all_images)}")
    
    return all_images


def visualize_images_static(all_images: dict, save_path: pathlib.Path = None):
    """
    Display images in a grid.
    
    Args:
        all_images: Dictionary mapping camera names to lists of images
        save_path: Path to save the visualization
    """
    if len(all_images) == 0:
        print("No images to display")
        return
    
    # If single camera, use old format for backward compatibility
    if len(all_images) == 1:
        camera_name = list(all_images.keys())[0]
        images = all_images[camera_name]
        _visualize_single_camera_static(images, camera_name, save_path)
    else:
        # Multiple cameras: show side by side
        _visualize_multi_camera_static(all_images, save_path)


def _visualize_single_camera_static(images: list, camera_name: str, save_path: pathlib.Path = None):
    """Display images from a single camera in a grid."""
    if len(images) == 0:
        print("No images to display")
        return
    
    # Calculate grid size
    num_images = len(images)
    cols = min(5, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
    if rows == 1:
        axes = axes.reshape(1, -1) if cols > 1 else [axes]
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        if rows == 1 and cols == 1:
            ax = axes
        elif rows == 1:
            ax = axes[col]
        elif cols == 1:
            ax = axes[row]
        else:
            ax = axes[row, col]
        
        # Calculate statistics
        img_mean = img.mean()
        img_max = img.max()
        img_min = img.min()
        
        ax.imshow(img)
        ax.set_title(f"{camera_name} - Frame {idx}\nmean={img_mean:.1f}, max={img_max}, min={img_min}")
        ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, rows * cols):
        row = idx // cols
        col = idx % cols
        if rows == 1 and cols == 1:
            continue
        elif rows == 1:
            axes[col].axis('off')
        elif cols == 1:
            axes[row].axis('off')
        else:
            axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def _visualize_multi_camera_static(all_images: dict, save_path: pathlib.Path = None):
    """Display images from multiple cameras side by side."""
    # Find the maximum number of images across all cameras
    max_frames = max(len(images) for images in all_images.values())
    num_cameras = len(all_images)
    
    # Create a grid: rows = frames, cols = cameras
    fig, axes = plt.subplots(max_frames, num_cameras, figsize=(5*num_cameras, 3*max_frames))
    
    if max_frames == 1:
        axes = axes.reshape(1, -1) if num_cameras > 1 else [axes]
    elif num_cameras == 1:
        axes = axes.reshape(-1, 1)
    
    camera_names = sorted(all_images.keys())
    
    for frame_idx in range(max_frames):
        for cam_idx, cam_name in enumerate(camera_names):
            images = all_images[cam_name]
            
            if max_frames == 1 and num_cameras == 1:
                ax = axes
            elif max_frames == 1:
                ax = axes[cam_idx]
            elif num_cameras == 1:
                ax = axes[frame_idx]
            else:
                ax = axes[frame_idx, cam_idx]
            
            if frame_idx < len(images):
                img = images[frame_idx]
                img_mean = img.mean()
                img_max = img.max()
                img_min = img.min()
                
                ax.imshow(img)
                ax.set_title(f"{cam_name}\nFrame {frame_idx}\nmean={img_mean:.1f}")
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, "N/A", ha='center', va='center')
            
            ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def visualize_images_animation(all_images: dict, save_path: pathlib.Path = None, fps: int = 10, camera_name: str = None):
    """
    Create animated visualization of images.
    
    Args:
        all_images: Dictionary mapping camera names to lists of images
        save_path: Path to save the animation
        fps: Frames per second
        camera_name: Specific camera to animate (None = first camera)
    """
    if len(all_images) == 0:
        print("No images to animate")
        return
    
    # Select camera
    if camera_name is None:
        camera_name = list(all_images.keys())[0]
    
    if camera_name not in all_images:
        print(f"Camera '{camera_name}' not found")
        return
    
    images = all_images[camera_name]
    if len(images) == 0:
        print("No images to animate")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Calculate statistics for first image
    img_mean = images[0].mean()
    img_max = images[0].max()
    img_min = images[0].min()
    
    im = ax.imshow(images[0])
    title = ax.set_title(f"{camera_name} - Frame 0 / {len(images)}\nmean={img_mean:.1f}, max={img_max}, min={img_min}")
    
    def update(frame):
        img = images[frame]
        im.set_array(img)
        
        img_mean = img.mean()
        img_max = img.max()
        img_min = img.min()
        
        title.set_text(f"{camera_name} - Frame {frame} / {len(images)}\nmean={img_mean:.1f}, max={img_max}, min={img_min}")
        return [im, title]
    
    anim = FuncAnimation(fig, update, frames=len(images), interval=1000/fps, blit=True, repeat=True)
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, writer='ffmpeg', fps=fps)
        print(f"Saved animation to {save_path}")
    else:
        plt.show()
    
    return anim


def main():
    parser = argparse.ArgumentParser(description="Visualize training data images")
    parser.add_argument(
        "--hdf5-path",
        type=str,
        required=True,
        help="Path to HDF5 trajectory file"
    )
    parser.add_argument(
        "--traj-idx",
        type=int,
        default=0,
        help="Trajectory index to visualize (default: 0)"
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=50,
        help="Maximum number of images to load (default: 50)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["static", "animation", "both"],
        default="static",
        help="Visualization mode (default: static)"
    )
    parser.add_argument(
        "--save-static",
        type=str,
        help="Path to save static visualization (PNG)"
    )
    parser.add_argument(
        "--save-animation",
        type=str,
        help="Path to save animation (MP4)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS for animation (default: 10)"
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Specific camera to visualize (front, left_wrist, right_wrist). Default: all cameras"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display plots (only save if paths provided)"
    )
    args = parser.parse_args()
    
    hdf5_path = pathlib.Path(args.hdf5_path)
    
    if not hdf5_path.exists():
        print(f"ERROR: File not found: {hdf5_path}")
        return
    
    # Load images
    print(f"Loading images from {hdf5_path}...")
    all_images = load_images_from_hdf5(hdf5_path, args.traj_idx, args.max_images, args.camera)
    
    if len(all_images) == 0:
        print("No images loaded. Check the file structure.")
        return
    
    # Print statistics for each camera
    print(f"\n{'='*60}")
    print(f"Image Statistics:")
    for cam_name, images in all_images.items():
        all_means = [img.mean() for img in images]
        all_maxs = [img.max() for img in images]
        all_mins = [img.min() for img in images]
        
        print(f"\n  Camera: {cam_name}")
        print(f"    Total images: {len(images)}")
        print(f"    Mean brightness: {np.mean(all_means):.2f} (min={np.min(all_means):.2f}, max={np.max(all_means):.2f})")
        print(f"    Max pixel value: {np.max(all_maxs)}")
        print(f"    Min pixel value: {np.min(all_mins)}")
        
        if np.mean(all_means) < 5.0:
            print(f"    ⚠️  WARNING: Images appear to be very dark or all black!")
        elif np.mean(all_means) > 50.0:
            print(f"    ✓ Images appear to have reasonable brightness")
    print(f"{'='*60}\n")
    
    # Visualize
    if args.mode in ["static", "both"]:
        save_path = pathlib.Path(args.save_static) if args.save_static else None
        if not args.no_display or save_path:
            visualize_images_static(all_images, save_path)
    
    if args.mode in ["animation", "both"]:
        save_path = pathlib.Path(args.save_animation) if args.save_animation else None
        if not args.no_display or save_path:
            # For animation, use first camera if multiple cameras
            camera_name = args.camera if args.camera else list(all_images.keys())[0]
            visualize_images_animation(all_images, save_path, args.fps, camera_name)


if __name__ == "__main__":
    main()
