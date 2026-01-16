"""
验证真机数据是否符合 LeRobot 标准格式
"""

import pathlib
import json
import pyarrow.parquet as pq

def verify_real_data(data_path: str = "real_data"):
    data_root = pathlib.Path(data_path)
    
    tasks = ["lift", "sort", "stack"]
    
    print("\n=== Verifying Real Robot Data Format ===\n")
    
    for task in tasks:
        task_path = data_root / task
        
        if not task_path.exists():
            print(f"⚠️  Task folder missing: {task}")
            continue
        
        try:
            print(f"✓ Checking '{task}' task...")
            
            # Check required directories
            required_dirs = ["data", "meta"]
            for d in required_dirs:
                if not (task_path / d).exists():
                    print(f"  ❌ Missing directory: {d}")
                    continue
            
            # Check required metadata files
            required_files = ["meta/info.json", "meta/stats.json"]
            for f in required_files:
                if not (task_path / f).exists():
                    print(f"  ❌ Missing file: {f}")
                    continue
            
            # Load and display metadata
            with open(task_path / "meta/info.json") as f:
                info = json.load(f)
            
            print(f"  - Episodes: {info.get('total_episodes', '?')}")
            print(f"  - Total frames: {info.get('total_frames', '?')}")
            print(f"  - Robot type: {info.get('robot_type', '?')}")
            print(f"  - FPS: {info.get('fps', '?')}")
            
            # Try to load first parquet file
            data_dir = task_path / "data" / "chunk-000"
            parquet_files = list(data_dir.glob("*.parquet"))
            
            if parquet_files:
                try:
                    first_file = sorted(parquet_files)[0]
                    table = pq.read_table(first_file)
                    print(f"  - Parquet columns: {table.column_names}")
                    print(f"  - Sample size: {len(table)} frames")
                    print(f"✅ Task '{task}' data format is valid!")
                except Exception as e:
                    print(f"  ⚠️  Could not read parquet file: {e}")
            else:
                print(f"  ❌ No parquet files found in {data_dir}")
            
            # Check if tasks.jsonl exists
            tasks_jsonl = task_path / "meta" / "tasks.jsonl"
            if tasks_jsonl.exists():
                print(f"  - tasks.jsonl: ✅ exists")
            else:
                print(f"  - tasks.jsonl: ❌ missing (will be created in next step)")
            
            print()
            
        except Exception as e:
            print(f"❌ Error checking '{task}': {e}")
            import traceback
            traceback.print_exc()
            print()

if __name__ == "__main__":
    verify_real_data()
    print("\nNext steps:")
    print("1. Run: uv run python scripts/prepare_real_data.py")
    print("   (This will create missing tasks.jsonl files)")
    print("2. Then proceed with training:")
    print("   uv run python scripts/train_real_data_lerobot.py --task lift")
