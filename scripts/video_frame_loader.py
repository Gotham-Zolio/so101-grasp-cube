"""
On-demand video frame loader with caching
Reads frames directly from MP4 files for minimal storage
"""
import sys
sys.path.insert(0, '/home/gotham/shared/so101-grasp-cube')

import pathlib
import cv2
import numpy as np
import pandas as pd
import glob
import json
from typing import Dict, Optional, Tuple
from functools import lru_cache
import hashlib

class VideoFrameLoader:
    """
    Efficiently load frames from MP4 videos on demand
    Caches open video handles to avoid repeated seeking
    """
    
    def __init__(self, task_dir: pathlib.Path, camera: str = 'observation.images.front'):
        self.task_dir = pathlib.Path(task_dir)
        self.camera = camera
        self.videos_dir = self.task_dir / 'videos' / camera / 'chunk-000'
        
        # Build frame_idx -> (video_file, local_frame_idx) mapping
        self.frame_to_video_mapping = self._build_mapping()
        
        # Cache for open video readers
        self.video_cache = {}  # video_path -> cv2.VideoCapture
        
        print(f"VideoFrameLoader initialized: {len(self.frame_to_video_mapping)} frames")
    
    def _build_mapping(self) -> Dict[int, Tuple[pathlib.Path, int]]:
        """
        Build mapping from global frame index to video file and local frame index
        Scans videos to determine frame boundaries
        """
        mapping = {}
        global_frame_idx = 0
        
        video_files = sorted(self.videos_dir.glob('*.mp4'))
        
        for video_file in video_files:
            # Get frame count for this video
            cap = cv2.VideoCapture(str(video_file))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Map local frame indices to global
            for local_idx in range(frame_count):
                mapping[global_frame_idx] = (video_file, local_idx)
                global_frame_idx += 1
        
        return mapping
    
    def _get_video_reader(self, video_path: pathlib.Path) -> cv2.VideoCapture:
        """Get or create cached video reader"""
        video_str = str(video_path)
        
        if video_str not in self.video_cache:
            cap = cv2.VideoCapture(video_str)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            self.video_cache[video_str] = cap
        
        return self.video_cache[video_str]
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get frame by global frame index
        Returns: (C, H, W) RGB image normalized to [0, 1]
        """
        if frame_idx not in self.frame_to_video_mapping:
            return None
        
        video_path, local_idx = self.frame_to_video_mapping[frame_idx]
        
        try:
            cap = self._get_video_reader(video_path)
            
            # Seek to frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, local_idx)
            ret, frame = cap.read()
            
            if not ret:
                return None
            
            # Convert BGR -> RGB and normalize
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2, 0, 1))  # HWC -> CHW
            frame = frame.astype(np.float32) / 255.0
            
            return frame
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
            return None
    
    def close(self):
        """Close all cached video readers"""
        for cap in self.video_cache.values():
            cap.release()
        self.video_cache.clear()
    
    def __del__(self):
        self.close()

def create_video_frame_index(task_dir: pathlib.Path) -> Dict[int, Tuple[str, int]]:
    """
    Create a JSON index mapping frame_idx -> (video_file, local_frame_idx)
    This allows fast lookup without opening all videos
    """
    index = {}
    global_frame_idx = 0
    
    videos_dir = task_dir / 'videos' / 'observation.images.front' / 'chunk-000'
    video_files = sorted(videos_dir.glob('*.mp4'))
    
    for video_file in video_files:
        cap = cv2.VideoCapture(str(video_file))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        for local_idx in range(frame_count):
            index[global_frame_idx] = {
                'video': video_file.name,
                'local_frame_idx': local_idx
            }
            global_frame_idx += 1
    
    # Save index
    index_file = task_dir / 'video_frame_index.json'
    with open(index_file, 'w') as f:
        json.dump(index, f)
    
    print(f"Created video frame index: {len(index)} frames -> {index_file}")
    return index

# Test the loader
if __name__ == '__main__':
    base_path = pathlib.Path('/home/gotham/shared/so101-grasp-cube/real_data')
    
    for task in ['lift']:
        print(f"\nTesting {task}...")
        task_dir = base_path / task
        
        # Create index
        index = create_video_frame_index(task_dir)
        
        # Initialize loader
        loader = VideoFrameLoader(task_dir)
        
        # Test loading a few frames
        print(f"Testing frame loading...")
        for idx in [0, 100, 500]:
            if idx in loader.frame_to_video_mapping:
                frame = loader.get_frame(idx)
                if frame is not None:
                    print(f"  Frame {idx}: shape={frame.shape}, dtype={frame.dtype}")
                else:
                    print(f"  Frame {idx}: FAILED")
        
        loader.close()
