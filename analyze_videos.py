"""
Analyze video structure and camera correspondence
"""
import sys
sys.path.insert(0, '/home/gotham/shared/so101-grasp-cube')

import pathlib
import cv2
import json
from collections import defaultdict

tasks = ['lift', 'sort', 'stack']
base_path = pathlib.Path('/home/gotham/shared/so101-grasp-cube/real_data')

for task in tasks:
    print(f"\n{'='*70}")
    print(f"TASK: {task.upper()}")
    print(f"{'='*70}")
    
    task_path = base_path / task
    videos_path = task_path / 'videos'
    
    # List all camera types
    camera_dirs = [d for d in videos_path.iterdir() if d.is_dir()]
    print(f"\nAvailable cameras: {[d.name for d in camera_dirs]}")
    
    for camera_dir in sorted(camera_dirs):
        camera_name = camera_dir.name
        print(f"\n  Camera: {camera_name}")
        
        # Find video files
        chunk_dir = camera_dir / 'chunk-000'
        if chunk_dir.exists():
            video_files = sorted(chunk_dir.glob('*.mp4'))
            print(f"    Video files: {len(video_files)}")
            
            for video_file in video_files[:1]:  # Check first video only
                print(f"    Checking: {video_file.name}")
                cap = cv2.VideoCapture(str(video_file))
                
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    print(f"      FPS: {fps}")
                    print(f"      Frame count: {frame_count}")
                    print(f"      Resolution: {width}x{height}")
                    
                    cap.release()
                else:
                    print(f"      Failed to open video")
    
    # Check meta info
    info_file = task_path / 'meta' / 'info.json'
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
        print(f"\n  Meta info:")
        print(f"    Episodes: {info.get('episodes', 'N/A')}")
        print(f"    Frames: {info.get('frames', 'N/A')}")
        print(f"    FPS: {info.get('fps', 'N/A')}")
        if 'robot_name' in info:
            print(f"    Robot: {info['robot_name']}")
