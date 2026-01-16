#!/usr/bin/env python3
"""
Debug script to test ffmpeg-based video frame loading
"""
import sys
import pathlib
import subprocess

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_ffmpeg_availability():
    """Check if ffmpeg and ffprobe are available"""
    print("Testing ffmpeg and ffprobe availability...")
    
    # Check ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=2)
        if result.returncode == 0:
            print("✓ ffmpeg is available")
        else:
            print("✗ ffmpeg returned error")
            return False
    except FileNotFoundError:
        print("✗ ffmpeg not found")
        return False
    
    # Check ffprobe
    try:
        result = subprocess.run(['ffprobe', '-version'], capture_output=True, timeout=2)
        if result.returncode == 0:
            print("✓ ffprobe is available")
        else:
            print("✗ ffprobe returned error")
            return False
    except FileNotFoundError:
        print("✗ ffprobe not found")
        return False
    
    return True


def test_video_frame_loader():
    """Test VideoFrameLoaderFFmpeg class"""
    print("\nTesting VideoFrameLoaderFFmpeg...")
    
    from scripts.video_frame_loader_ffmpeg import VideoFrameLoader
    
    # Test with lift task
    task_dir = project_root / "real_data" / "lift"
    
    if not task_dir.exists():
        print(f"⚠ Task directory not found: {task_dir}")
        return False
    
    try:
        loader = VideoFrameLoader(task_dir, camera='observation.images.front')
        print(f"✓ VideoFrameLoader created: {len(loader)} frames")
        
        # Try to get first frame
        frame = loader.get_frame(0)
        if frame is not None:
            print(f"✓ Got first frame: shape={frame.shape}, dtype={frame.dtype}, min={frame.min():.4f}, max={frame.max():.4f}")
            return True
        else:
            print("✗ Failed to get first frame (returned None)")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_parquet_dataset():
    """Test ParquetRealDataset with image loading"""
    print("\nTesting ParquetRealDataset with images...")
    
    from scripts.train_real_data_lerobot import ParquetRealDataset
    
    task_dir = project_root / "real_data" / "lift"
    
    if not task_dir.exists():
        print(f"⚠ Task directory not found: {task_dir}")
        return False
    
    try:
        dataset = ParquetRealDataset(task_dir, load_images=True)
        print(f"✓ ParquetRealDataset created: {len(dataset)} samples")
        
        # Get first sample
        sample = dataset[0]
        print(f"✓ Got first sample with keys: {sample.keys()}")
        
        # Check image
        if "observation.images.front" in sample:
            img = sample["observation.images.front"]
            print(f"✓ Image present: shape={img.shape}, dtype={img.dtype}, min={img.min():.4f}, max={img.max():.4f}")
        else:
            print("✗ Image not in sample")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("FFmpeg-based Video Frame Loading Tests")
    print("="*60 + "\n")
    
    all_passed = True
    
    # Test 1: Check ffmpeg/ffprobe
    if not test_ffmpeg_availability():
        all_passed = False
        print("\n⚠ ffmpeg/ffprobe not available - video loading will fail")
        sys.exit(1)
    
    # Test 2: VideoFrameLoader
    if not test_video_frame_loader():
        all_passed = False
    
    # Test 3: ParquetRealDataset with images
    if not test_parquet_dataset():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*60 + "\n")
    
    sys.exit(0 if all_passed else 1)
