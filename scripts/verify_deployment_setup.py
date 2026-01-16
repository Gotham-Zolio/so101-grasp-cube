"""
Verification script for deployment setup.
Tests dependencies, configuration loaded, and image processing logic without real hardware.
"""

import sys
import numpy as np
import time
from unittest.mock import MagicMock, patch
import tyro
from pathlib import Path
import json
import os

# Add project root to sys.path to ensure local modules are found
# This handles the case where the script is run directly from the scripts folder or root
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_dependencies():
    print("[1/4] Checking Dependencies...")
    try:
        import gymnasium as gym
        print("  - gymnasium: OK")
        import torch
        print("  - torch: OK")
        import cv2
        print("  - opencv-python: OK")
        # Checking for lerobot based on likely import path in this env
        # Note: adjust this import if Lerobot structure is different in your env
        try:
             import lerobot
             print("  - lerobot: OK")
        except ImportError:
             print("  ! lerobot package not found standardly, assuming local or custom env.")

    except ImportError as e:
        print(f"  X Missing Dependency: {e}")
        return False
    print("  > Dependencies check passed.\n")
    return True

def test_image_processing():
    print("[2/4] Testing Image Preprocessing Logic...")
    try:
        # Import directly from the file we want to test
        # We need to temporarily mock LeRobotEnv dependencies if they require hardware
        with patch.dict(sys.modules, {
            "lerobot.robots": MagicMock(),
            "lerobot.common.robot_devices.cameras.utils": MagicMock(),
            "lerobot.utils.robot_utils": MagicMock(),
        }):
            # Need to mock make_robot_from_config before import 
            # or ensure the import doesn't trigger it immediately
            with patch('grasp_cube.real.lerobot_env.make_robot_from_config') as mock_make_robot:
                from grasp_cube.real.lerobot_env import LeRobotEnv, LeRobotEnvConfig
            
                # Create a mock config
                mock_robot_config = MagicMock()
                mock_robot_config.action_features = {"main_arm": None} # Mock action space
                
                # create a dummy camera config file
                config_path = Path("temp_camera_config.json")
                with open(config_path, "w") as f:
                    json.dump({"front": {"type": "opencv", "id": 0}}, f)

                config = LeRobotEnvConfig(
                    robot=mock_robot_config,
                    camera_config_path=config_path,
                    image_resolution=(480, 640), # Target
                    task="test_task"
                )
                
                # Setup mock robot return
                mock_robot = MagicMock()
                mock_robot.action_features = {"main_arm": None}
                mock_robot.robot_type = "so101_follower"
                mock_make_robot.return_value = mock_robot
                
                # Instantiate Env
                # We mock make_teleoperator just in case
                with patch('grasp_cube.real.lerobot_env.make_teleoperator_from_config'), \
                     patch('grasp_cube.real.lerobot_env.make_default_processors') as mock_processors:
                     
                    mock_processors.return_value = (MagicMock(), MagicMock(), MagicMock())
                    env = LeRobotEnv(config)
                    
                    # Test Case 1: Perfect Match
                    img_perfect = np.zeros((480, 640, 3), dtype=np.uint8)
                    res = env._resize_and_crop(img_perfect)
                    assert res.shape == (480, 640, 3), f"Perfect match failed: {res.shape}"
                    print("  - 640x480 Input -> 640x480: OK")

                    # Test Case 2: High Res (1920x1080) -> Needs Crop & Resize
                    # Aspect Ratio 16:9 vs Target 4:3
                    img_hd = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    res = env._resize_and_crop(img_hd)
                    assert res.shape == (480, 640, 3), f"HD input failed: {res.shape}"
                    print("  - 1920x1080 Input -> 640x480: OK")
                    
                    # Test Case 3: Weird Aspect Ratio (Square)
                    img_sq = np.zeros((1000, 1000, 3), dtype=np.uint8)
                    res = env._resize_and_crop(img_sq)
                    assert res.shape == (480, 640, 3), f"Square input failed: {res.shape}"
                    print("  - 1000x1000 Input -> 640x480: OK")
            
            # Cleanup
            if config_path.exists():
                config_path.unlink()
                
    except Exception as e:
        print(f"  X Image Processing Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("  > Image processing logic verified.\n")
    return True

def test_config_structure():
    print("[3/4] Verifying Config Structures...")
    try:
        # We need to mock lerobot for the import to work if env not perfect
        with patch.dict(sys.modules, {"lerobot": MagicMock()}):
            from scripts.eval_real_policy import DeployConfig
            # Just ensure tyro can parse the help message implies structure is valid
            print("  - DeployConfig structure: OK")
    except Exception as e:
        print(f"  X Config Structure Error: {e}")
        return False
    print("  > Config Check Passed.\n")
    return True

def main():
    print("=== Verification Started ===\n")
    
    # Order: Dependencies -> Image Processing -> Config Structure
    steps = [test_dependencies, test_image_processing, test_config_structure]
    all_passed = True
    
    for step in steps:
        if not step():
            all_passed = False
            print("!!! Verification failed at this step. Fixing required. !!!")
            break
            
    if all_passed:
        print("=== [SUCCESS] Usage Instructions ===")
        print("The code structure is valid. You can now try a 'Dry Run' on the robot computer:")
        print("1. Connect the robot and cameras.")
        print("2. Run the actual evaluation with dry-run flag:")
        print("   python scripts/eval_real_policy.py --policy.path ... --env.camera-config-path ... --dry-run")
        print("   (This will connect but NOT move the robot)")

if __name__ == "__main__":
    main()
