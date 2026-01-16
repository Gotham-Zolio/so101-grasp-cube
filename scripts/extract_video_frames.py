"""
Extract frames from MP4 videos and save as parquet
Handles AV1 encoded videos by using ffmpeg
"""
import subprocess
import sys
import pathlib
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional
import glob

def extract_frames_ffmpeg(video_path: str, output_dir: pathlib.Path, start_idx: int = 0) -> int:
    """
    Extract frames from video using ffmpeg
    Returns: number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use ffmpeg to extract frames
    # %06d creates zero-padded frame numbers like 000000.png, 000001.png, etc.
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-q:v', '2',  # Quality (2 is highest)
        '-f', 'image2',
        str(output_dir / 'frame_%06d.png')
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        return 0
    
    # Count extracted frames
    frame_files = sorted(output_dir.glob('frame_*.png'))
    return len(frame_files)

def get_frame_count(video_path: str) -> int:
    """Get frame count using ffmpeg"""
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames',
        '-of', 'default=noprint_wrappers=1:nokey=1:noprint_wrappers=1',
        video_path
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(result.stdout.strip())
    except:
        return 0

print("="*70)
print("VIDEO EXTRACTION PIPELINE")
print("="*70)
print("\nThis script extracts frames from AV1-encoded MP4 videos")
print("Requires: ffmpeg, ffprobe")
print("\nNote: Extraction will create temporary frame directories")
print("After verification, these can be removed to save space")

tasks = ['lift', 'sort', 'stack']
base_path = pathlib.Path('/home/gotham/shared/so101-grasp-cube/real_data')

for task in tasks:
    print(f"\n{'='*70}")
    print(f"TASK: {task.upper()}")
    print(f"{'='*70}")
    
    task_path = base_path / task
    videos_path = task_path / 'videos'
    
    # Use front camera only
    camera_name = 'observation.images.front'
    camera_path = videos_path / camera_name
    
    if not camera_path.exists():
        print(f"  ✗ Camera {camera_name} not found")
        continue
    
    print(f"  Camera: {camera_name}")
    
    # Find chunk directories
    for chunk_dir in sorted(camera_path.iterdir()):
        if not chunk_dir.is_dir():
            continue
        
        chunk_name = chunk_dir.name
        print(f"    Chunk: {chunk_name}")
        
        video_files = sorted(chunk_dir.glob('*.mp4'))
        for video_idx, video_file in enumerate(video_files):
            print(f"      Video {video_idx}: {video_file.name}", end='')
            
            # Get frame count
            frame_count = get_frame_count(str(video_file))
            print(f" ({frame_count} frames)")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)
print("""
1. Install ffmpeg if not available:
   apt-get install ffmpeg
   
2. Run extraction for each task:
   python extract_video_frames.py --task lift --camera front
   
3. Frames will be saved to: real_data/{task}/frames/{camera}/{chunk}/
   
4. Then integrate frames into parquet via a new dataset loader

Choose to proceed? (y/n)
""")
