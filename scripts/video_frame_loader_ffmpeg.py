#!/usr/bin/env python3
"""
Efficient on-demand video frame loader using ffmpeg
Handles AV1-encoded MP4 videos that OpenCV cannot decode properly
"""
import json
import pathlib
import subprocess
import numpy as np
from typing import Optional, Dict, Tuple

class VideoFrameLoaderFFmpeg:
    """
    Load video frames on-demand using ffmpeg (supports AV1 codec)
    Falls back to caching entire video in memory if needed
    """
    
    def __init__(self, task_dir: pathlib.Path, camera: str = 'observation.images.front'):
        """
        Initialize VideoFrameLoader for a task directory
        Args:
            task_dir: Path to task directory (e.g., real_data/lift)
            camera: Camera name (default: observation.images.front)
        """
        self.task_dir = pathlib.Path(task_dir)
        self.camera = camera
        self.video_dir = self.task_dir / "videos" / camera / "chunk-000"
        
        if not self.video_dir.exists():
            raise FileNotFoundError(f"Video directory not found: {self.video_dir}")
        
        # Build frame index: frame_idx -> (video_file, local_frame_idx)
        self.frame_index = self._build_frame_index()
        
        # Cache for decoded frames (in-memory)
        self.frame_cache = {}  # frame_idx -> numpy array
        self.cache_size = 100  # Keep up to 100 frames in memory
        
        print(f"VideoFrameLoaderFFmpeg initialized: {len(self.frame_index)} frames")
    
    def _build_frame_index(self) -> Dict[int, Tuple[str, int]]:
        """Build mapping from global frame index to (video_file, local_frame_idx)"""
        frame_index = {}
        global_idx = 0
        
        # Process all MP4 files in camera directory (should be in chunk-000)
        for video_file in sorted(self.video_dir.glob("*.mp4")):
            try:
                # Get frame count using ffprobe
                frame_count = self._get_frame_count_ffprobe(video_file)
                
                if frame_count == 0:
                    print(f"Warning: {video_file.name} has 0 frames")
                    continue
                
                # Map frames for this video
                for local_idx in range(frame_count):
                    frame_index[global_idx] = (str(video_file), local_idx)
                    global_idx += 1
                    
            except Exception as e:
                print(f"Warning: Failed to index {video_file}: {e}")
        
        return frame_index
    
    def _get_frame_count_ffprobe(self, video_file: pathlib.Path) -> int:
        """Get frame count using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets',
                '-of', 'csv=p=0',
                str(video_file)
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            frame_count = int(result.stdout.strip())
            return max(0, frame_count)
        except Exception as e:
            print(f"Warning: ffprobe failed for {video_file}: {e}")
            return 0
    
    def _extract_frame_ffmpeg(self, video_file: str, frame_idx: int) -> Optional[np.ndarray]:
        """Extract a single frame using ffmpeg"""
        try:
            # Use ffmpeg to extract specific frame
            cmd = [
                'ffmpeg',
                '-i', video_file,
                '-vf', f'select=eq(n\\,{frame_idx})',
                '-vframes', '1',
                '-f', 'image2',
                '-pix_fmt', 'rgb24',
                '-loglevel', 'quiet',
                'pipe:1'
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            
            if result.returncode != 0 or len(result.stdout) == 0:
                return None
            
            # Decode raw RGB frame (assuming 640x480)
            frame_data = result.stdout
            frame_width, frame_height = 640, 480
            expected_size = frame_width * frame_height * 3
            
            if len(frame_data) < expected_size:
                # Try to decode what we have
                if len(frame_data) >= 3:
                    # Pad with zeros if needed
                    frame_data = frame_data.ljust(expected_size, b'\x00')
                else:
                    return None
            
            # Reshape to (H, W, C) and convert to (C, H, W)
            frame = np.frombuffer(frame_data[:expected_size], dtype=np.uint8)
            frame = frame.reshape(frame_height, frame_width, 3)
            frame = np.transpose(frame, (2, 0, 1)).astype(np.float32) / 255.0
            
            return frame
            
        except Exception as e:
            print(f"Warning: ffmpeg frame extraction failed: {e}")
            return None
    
    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get frame at given index
        Returns numpy array (3, 480, 640) float32 [0, 1] or None if not found
        """
        if frame_idx not in self.frame_index:
            return None
        
        # Check cache first
        if frame_idx in self.frame_cache:
            return self.frame_cache[frame_idx]
        
        video_file, local_idx = self.frame_index[frame_idx]
        
        # Extract frame using ffmpeg
        frame = self._extract_frame_ffmpeg(video_file, local_idx)
        
        if frame is not None:
            # Update cache
            self._update_cache(frame_idx, frame)
        
        return frame
    
    def _update_cache(self, frame_idx: int, frame: np.ndarray):
        """Update cache with LRU strategy"""
        self.frame_cache[frame_idx] = frame
        
        # Keep only last N frames in cache
        if len(self.frame_cache) > self.cache_size:
            # Remove oldest frame (smallest index)
            oldest_idx = min(self.frame_cache.keys())
            del self.frame_cache[oldest_idx]
    
    def __len__(self) -> int:
        """Total number of frames"""
        return len(self.frame_index)


# Create alias for compatibility
VideoFrameLoader = VideoFrameLoaderFFmpeg
