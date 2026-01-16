"""
修复真机数据格式以符合 LeRobot 新版本要求
"""

import pathlib
import json
import jsonlines
from typing import Dict, Any

def create_tasks_jsonl(data_path: str = "real_data"):
    """
    为每个任务创建缺失的 tasks.jsonl 文件
    
    LeRobot 3.1+ 要求每个数据集都有 tasks.jsonl，其中记录数据集中包含的任务列表
    """
    data_root = pathlib.Path(data_path)
    
    tasks_list = ["lift", "sort", "stack"]
    
    for task in tasks_list:
        task_path = data_root / task
        meta_path = task_path / "meta"
        tasks_jsonl_path = meta_path / "tasks.jsonl"
        
        if not task_path.exists():
            print(f"⚠️  Task folder missing: {task}")
            continue
        
        # 读取 info.json 获取任务描述
        info_path = meta_path / "info.json"
        if not info_path.exists():
            print(f"⚠️  info.json missing for task {task}")
            continue
        
        with open(info_path, 'r') as f:
            info = json.load(f)
        
        # 创建 tasks.jsonl
        # 根据任务生成合理的描述
        task_descriptions = {
            "lift": "Pick up the red cube and lift it.",
            "sort": "Move the red cube to the left region and the green cube to the right region.",
            "stack": "Stack the red cube on top of the green cube.",
            "pick": "Pick up the red cube.",
        }
        
        task_data = {
            "task_index": 0,
            "task": task_descriptions.get(task, f"Complete {task} task"),
            "robot_type": info.get("robot_type", "so101_follower"),
        }
        
        # 写入 tasks.jsonl
        try:
            with jsonlines.open(tasks_jsonl_path, mode="w") as writer:
                writer.write(task_data)
            print(f"✅ Created tasks.jsonl for '{task}'")
        except Exception as e:
            print(f"❌ Failed to create tasks.jsonl for '{task}': {e}")
            continue

def verify_data_structure(data_path: str = "real_data"):
    """验证数据结构完整性"""
    data_root = pathlib.Path(data_path)
    
    tasks_list = ["lift", "sort", "stack"]
    required_files = [
        "meta/info.json",
        "meta/stats.json",
        "meta/tasks.jsonl",  # 新增
        "data",
    ]
    
    print("\n=== Data Structure Verification ===\n")
    
    for task in tasks_list:
        task_path = data_root / task
        print(f"Checking '{task}' task:")
        
        all_ok = True
        for f in required_files:
            full_path = task_path / f
            exists = full_path.exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {f}")
            all_ok = all_ok and exists
        
        if all_ok:
            print(f"✅ '{task}' is ready for training!\n")
        else:
            print(f"⚠️  '{task}' has missing files\n")

if __name__ == "__main__":
    print("=== Preparing Real Data for LeRobot Training ===\n")
    
    # Step 1: Create missing tasks.jsonl files
    print("[1/2] Creating tasks.jsonl files...\n")
    create_tasks_jsonl()
    
    # Step 2: Verify structure
    print("\n[2/2] Verifying data structure...\n")
    verify_data_structure()
    
    print("=" * 50)
    print("✅ Data preparation completed!")
    print("\nYou can now proceed with training:")
    print("  uv run python scripts/train_real_data_lerobot.py --task lift")
