#!/usr/bin/env python3
"""
一键训练所有 ACT 模型

这个脚本依次训练三个任务（lift, sort, stack）的 ACT 模型。

用法：
    python scripts/train_all_act_models.py
    
或指定自定义参数：
    python scripts/train_all_act_models.py --epochs 200 --batch-size 16
"""

import argparse
import subprocess
import sys
import pathlib
from typing import List
import time


def run_training(
    task: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    num_workers: int,
    data_dir: pathlib.Path,
    output_base_dir: pathlib.Path,
) -> bool:
    """运行单个任务的训练"""
    
    print(f"\n{'='*70}")
    print(f"Starting training for task: {task.upper()}")
    print(f"{'='*70}")
    
    output_dir = output_base_dir / f"{task}_act"
    
    cmd = [
        sys.executable,
        "scripts/train_act_real_data.py",
        "--task", task,
        "--data-dir", str(data_dir),
        "--output-dir", str(output_dir),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--device", device,
        "--num-workers", str(num_workers),
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ Training completed for {task.upper()}")
        print(f"  Output directory: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Training failed for {task.upper()}")
        print(f"  Error code: {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Could not find training script")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train ACT models for all tasks (lift, sort, stack)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs for each task",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--data-dir",
        type=pathlib.Path,
        default=pathlib.Path("real_data"),
        help="Root directory of real data",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints"),
        help="Base output directory for all checkpoints",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["lift", "sort", "stack"],
        choices=["lift", "sort", "stack"],
        help="Tasks to train (default: all)",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Continue training next task even if one fails",
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"ACT Model Training Suite")
    print(f"{'='*70}")
    print(f"Tasks to train: {', '.join(args.tasks)}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Epochs per task: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print(f"Workers: {args.num_workers}")
    
    # 检查数据目录
    if not args.data_dir.exists():
        print(f"\n✗ Error: Data directory not found: {args.data_dir}")
        return 1
    
    # 检查至少有一个任务数据
    found_any = False
    for task in args.tasks:
        task_dir = args.data_dir / task
        if task_dir.exists():
            found_any = True
            print(f"  ✓ Found data for {task}")
        else:
            print(f"  ⚠ No data found for {task}")
    
    if not found_any:
        print(f"\n✗ Error: No task data found in {args.data_dir}")
        return 1
    
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 开始训练
    print(f"\n{'='*70}\n")
    
    results = {}
    start_time = time.time()
    
    for task in args.tasks:
        task_dir = args.data_dir / task
        if not task_dir.exists():
            print(f"\n⊘ Skipping {task.upper()} (data not found)")
            results[task] = "SKIPPED"
            continue
        
        task_start = time.time()
        success = run_training(
            task=task,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            output_base_dir=args.output_dir,
        )
        task_time = time.time() - task_start
        
        results[task] = "SUCCESS" if success else "FAILED"
        
        print(f"Training time for {task}: {task_time:.1f}s ({task_time/60:.1f}m)")
        
        if not success and not args.skip_failed:
            print(f"\n✗ Training failed. Stopping.")
            return 1
    
    # 总结
    total_time = time.time() - start_time
    
    print(f"\n{'='*70}")
    print(f"Training Summary")
    print(f"{'='*70}")
    
    for task, status in results.items():
        symbol = "✓" if status == "SUCCESS" else "⚠" if status == "SKIPPED" else "✗"
        print(f"{symbol} {task.upper()}: {status}")
    
    successful = sum(1 for s in results.values() if s == "SUCCESS")
    print(f"\nSuccessful tasks: {successful}/{len([s for s in results.values() if s != 'SKIPPED'])}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"{'='*70}\n")
    
    # 输出检查点位置
    print("Trained models location:")
    for task in args.tasks:
        if results.get(task) == "SUCCESS":
            checkpoint = args.output_dir / f"{task}_act" / "checkpoint-best"
            print(f"  {task}: {checkpoint}")
    
    print("\nNext steps:")
    print("1. Evaluate models:")
    print("   python scripts/eval_sim_policy.py --checkpoint checkpoints/lift_act/checkpoint-best")
    print("2. Deploy to real robot:")
    print("   python serve_act_policy.py --checkpoint checkpoints/lift_act/checkpoint-best")
    
    return 0 if successful > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
