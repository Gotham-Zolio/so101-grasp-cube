"""
Extract frames from AV1-encoded MP4 videos and integrate into parquet
Creates a mapping of frame_index -> image for training
"""
import sys
sys.path.insert(0, '/home/gotham/shared/so101-grasp-cube')

import subprocess
import pathlib
import numpy as np
import pandas as pd
import cv2
import json
from typing import Dict
import glob
from tqdm import tqdm
import pickle

def get_frame_count_ffmpeg(video_path: str) -> int:
    """Get frame count using ffprobe"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        count = int(result.stdout.strip())
        return count if count > 0 else 0
    except:
        return 0

def extract_frames_ffmpeg(video_path: str, output_dir: pathlib.Path) -> int:
    """
    Extract frames from video using ffmpeg
    Returns number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use ffmpeg to extract frames as PNG
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-q:v', '2',  # Quality level 2 (best)
        '-f', 'image2',
        '-start_number', '0',
        str(output_dir / 'frame_%06d.png')
    ]
    
    result = subprocess.run(cmd, capture_output=True, stderr=subprocess.DEVNULL)
    
    if result.returncode != 0:
        return 0
    
    # Count extracted frames
    frames = sorted(output_dir.glob('frame_*.png'))
    return len(frames)

def load_frame_as_array(frame_path: pathlib.Path) -> np.ndarray:
    """Load PNG frame as numpy array (RGB, float32)"""
    img = cv2.imread(str(frame_path))  # BGR
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    return img.astype(np.float32) / 255.0  # Normalize to [0, 1]

def process_video_file(
    video_path: pathlib.Path,
    parquet_df: pd.DataFrame,
    video_start_frame_idx: int,
    camera_name: str,
    frames_dir: pathlib.Path
) -> Dict[int, np.ndarray]:
    """
    Extract frames from video and create frame_idx -> image mapping
    """
    print(f"    Extracting: {video_path.name}")
    
    # Extract frames to temp directory
    temp_frames_dir = frames_dir / f"temp_{video_path.stem}"
    frame_count = extract_frames_ffmpeg(str(video_path), temp_frames_dir)
    
    if frame_count == 0:
        print(f"      ✗ Failed to extract frames")
        return {}
    
    print(f"      ✓ Extracted {frame_count} frames")
    
    # Load frames and create mapping
    frame_mapping = {}
    frame_files = sorted(temp_frames_dir.glob('frame_*.png'))
    
    for local_frame_idx, frame_file in enumerate(frame_files):
        global_frame_idx = video_start_frame_idx + local_frame_idx
        
        # Skip if this frame index is not in our parquet
        if global_frame_idx not in parquet_df.index:
            continue
        
        # Load frame
        frame_array = load_frame_as_array(frame_file)
        if frame_array is not None:
            frame_mapping[global_frame_idx] = frame_array
    
    print(f"      ✓ Mapped {len(frame_mapping)} frames to parquet indices")
    
    return frame_mapping

def create_frame_index_mapping(task: str, base_path: pathlib.Path):
    """
    Create mapping from frame_index in parquet to actual image arrays
    Saves as pickle for fast loading
    """
    task_path = base_path / task
    data_path = task_path / 'data'
    videos_path = task_path / 'videos'
    frames_dir = task_path / 'frames'
    frames_dir.mkdir(exist_ok=True)
    
    # Load all parquet data
    print(f"\n{'='*70}")
    print(f"Processing {task.upper()}")
    print(f"{'='*70}")
    
    parquet_files = sorted(glob.glob(str(data_path / 'chunk-*' / '*.parquet')))
    all_dfs = []
    for pf in parquet_files:
        df = pd.read_parquet(pf)
        all_dfs.append(df)
    
    parquet_df = pd.concat(all_dfs, ignore_index=False)
    print(f"\nLoaded {len(parquet_df)} frames from parquet")
    
    # Process front camera videos
    camera_name = 'observation.images.front'
    camera_path = videos_path / camera_name
    
    all_frame_mappings = {}
    
    if camera_path.exists():
        print(f"\nProcessing {camera_name}")
        
        for chunk_dir in sorted(camera_path.iterdir()):
            if not chunk_dir.is_dir():
                continue
            
            chunk_name = chunk_dir.name
            print(f"  Chunk: {chunk_name}")
            
            video_files = sorted(chunk_dir.glob('*.mp4'))
            
            for video_idx, video_file in enumerate(video_files):
                # Determine start frame index for this video
                video_start_frame_idx = video_idx * 10000  # Rough estimate
                
                # Try to get exact count from ffprobe
                frame_count = get_frame_count_ffmpeg(str(video_file))
                if frame_count == 0:
                    print(f"    ✗ {video_file.name}: Could not determine frame count")
                    continue
                
                print(f"    Video {video_idx}: {video_file.name} ({frame_count} frames)")
                
                frame_mapping = process_video_file(
                    video_file,
                    parquet_df,
                    video_start_frame_idx,
                    camera_name,
                    frames_dir
                )
                all_frame_mappings.update(frame_mapping)
    
    # Save frame mappings as pickle for fast loading
    mapping_file = task_path / f'frame_index_to_image_{camera_name.replace(".", "_")}.pkl'
    print(f"\n  Saving frame mapping to {mapping_file.name}")
    
    # Create a smaller index-only mapping (actual arrays loaded on demand)
    frame_index_mapping = {
        'frame_indices': list(all_frame_mappings.keys()),
        'camera': camera_name,
        'frames_dir': str(frames_dir),
    }
    
    with open(mapping_file, 'wb') as f:
        pickle.dump(frame_index_mapping, f)
    
    print(f"  ✓ Saved {len(all_frame_mappings)} frame mappings")
    
    return mapping_file

if __name__ == '__main__':
    base_path = pathlib.Path('/home/gotham/shared/so101-grasp-cube/real_data')
    
    # Check ffmpeg
    result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
    if result.returncode != 0:
        print("ERROR: ffmpeg not found. Install with: apt-get install ffmpeg")
        sys.exit(1)
    
    print("✓ ffmpeg found")
    
    # Process each task
    for task in ['lift', 'sort', 'stack']:
        try:
            mapping_file = create_frame_index_mapping(task, base_path)
            print(f"✓ {task} frame mapping created: {mapping_file}")
        except Exception as e:
            print(f"✗ {task} processing failed: {e}")
    
    print(f"\n{'='*70}")
    print("Frame extraction complete!")
    print("="*70)
