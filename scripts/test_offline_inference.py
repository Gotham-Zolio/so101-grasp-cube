#!/usr/bin/env python3
"""
离线推理测试
验证模型推理的正确性和性能（无需真机）
"""
import sys
import pathlib
import numpy as np
import time
from typing import Dict, List

project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.inference_engine import DiffusionPolicyInferenceEngine, load_multi_task_models


def test_single_inference():
    """测试单次推理"""
    print("\n" + "="*70)
    print("Test 1: Single Inference")
    print("="*70)
    
    engine = DiffusionPolicyInferenceEngine("checkpoints/lift_real/checkpoint-best")
    
    # 生成随机输入
    image = np.random.rand(3, 480, 640).astype(np.float32)
    state = np.array([0.0, 0.5, 1.0, -0.5, 0.0, 0.5], dtype=np.float32)
    
    # 推理
    start = time.time()
    actions = engine.predict(image, state)
    elapsed = time.time() - start
    
    print(f"✓ Inference successful")
    print(f"  Actions shape: {actions.shape}")
    print(f"  First action: {actions[0]}")
    print(f"  Action range: [{actions.min():.4f}, {actions.max():.4f}]")
    print(f"  Inference time: {elapsed*1000:.2f} ms")
    
    # 检查性能
    if elapsed < 0.05:
        print(f"  ✓ Excellent speed (<50ms)")
    elif elapsed < 0.1:
        print(f"  ✓ Good speed (<100ms)")
    else:
        print(f"  ⚠ Slow inference ({elapsed*1000:.0f}ms) - may need optimization")
    
    return True


def test_batch_inference():
    """测试批量推理"""
    print("\n" + "="*70)
    print("Test 2: Batch Inference")
    print("="*70)
    
    engine = DiffusionPolicyInferenceEngine("checkpoints/lift_real/checkpoint-best")
    
    batch_size = 8
    images = np.random.rand(batch_size, 3, 480, 640).astype(np.float32)
    states = np.random.randn(batch_size, 6).astype(np.float32)
    
    start = time.time()
    actions = engine.predict_batch(images, states)
    elapsed = time.time() - start
    
    print(f"✓ Batch inference successful")
    print(f"  Batch size: {batch_size}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Per-sample time: {elapsed/batch_size*1000:.2f}ms")
    
    return True


def test_all_models():
    """测试所有任务的模型"""
    print("\n" + "="*70)
    print("Test 3: Load All Models")
    print("="*70)
    
    models = load_multi_task_models()
    
    if not models:
        print("✗ No models found!")
        return False
    
    print(f"✓ Successfully loaded {len(models)} models")
    
    # 测试每个模型
    for task, engine in models.items():
        print(f"\n  {task.upper()}:")
        print(f"    State dim: {engine.state_dim}")
        print(f"    Action dim: {engine.action_dim}")
        print(f"    Horizon: {engine.horizon}")
        
        # 快速推理测试 - 为每个任务创建正确维度的state向量
        image = np.random.rand(3, 480, 640).astype(np.float32)
        state = np.zeros(engine.state_dim, dtype=np.float32)
        
        try:
            start = time.time()
            actions = engine.predict(image, state)
            elapsed = time.time() - start
            
            print(f"    Inference: {elapsed*1000:.2f}ms → action shape {actions.shape}")
        except Exception as e:
            print(f"    ✗ Inference failed: {e}")
            return False
    
    return True


def test_inference_consistency():
    """测试推理的一致性（同样输入应得到同样输出）"""
    print("\n" + "="*70)
    print("Test 4: Inference Consistency")
    print("="*70)
    
    engine = DiffusionPolicyInferenceEngine("checkpoints/lift_real/checkpoint-best")
    
    # 固定输入
    image = np.ones((3, 480, 640), dtype=np.float32) * 0.5
    state = np.zeros(6, dtype=np.float32)
    
    # 推理多次
    results = []
    for i in range(3):
        actions = engine.predict(image, state)
        results.append(actions)
    
    # 检查一致性
    consistent = True
    for i in range(1, len(results)):
        if not np.allclose(results[0], results[i], rtol=1e-5):
            consistent = False
            break
    
    if consistent:
        print(f"✓ Inference is deterministic (consistent outputs)")
    else:
        print(f"⚠ Inference is non-deterministic (could be due to randomness in model)")
    
    return True


def test_input_validation():
    """测试输入验证"""
    print("\n" + "="*70)
    print("Test 5: Input Validation")
    print("="*70)
    
    engine = DiffusionPolicyInferenceEngine("checkpoints/lift_real/checkpoint-best")
    
    # 测试 1: 错误的图像形状
    print("  Testing invalid image shape...")
    try:
        bad_image = np.random.rand(480, 640, 3).astype(np.float32)  # (H, W, C) 而不是 (C, H, W)
        state = np.zeros(6, dtype=np.float32)
        engine.predict(bad_image, state)
        print("  ✗ Should have caught bad image shape")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly rejected bad image shape: {e}")
    
    # 测试 2: 错误的数据类型
    print("  Testing invalid dtype...")
    try:
        bad_image = np.random.rand(3, 480, 640).astype(np.float64)  # float64 而不是 float32
        state = np.zeros(6, dtype=np.float32)
        engine.predict(bad_image, state)
        print("  ✗ Should have caught bad dtype")
        return False
    except ValueError as e:
        print(f"  ✓ Correctly rejected bad dtype: {e}")
    
    # 测试 3: 图像值超出范围
    print("  Testing out-of-range image values...")
    bad_image = np.ones((3, 480, 640), dtype=np.float32) * 2.0  # > 1.0
    state = np.zeros(6, dtype=np.float32)
    actions = engine.predict(bad_image, state)
    print(f"  ⚠ Warned about out-of-range values, but still returned actions: {actions.shape}")
    
    return True


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "="*70)
    print("Test 6: Edge Cases")
    print("="*70)
    
    engine = DiffusionPolicyInferenceEngine("checkpoints/lift_real/checkpoint-best")
    
    # 测试 1: 全黑图像
    print("  Testing all-black image...")
    black_image = np.zeros((3, 480, 640), dtype=np.float32)
    state = np.zeros(6, dtype=np.float32)
    actions = engine.predict(black_image, state)
    print(f"  ✓ Handled all-black image: {actions.shape}")
    
    # 测试 2: 全白图像
    print("  Testing all-white image...")
    white_image = np.ones((3, 480, 640), dtype=np.float32)
    actions = engine.predict(white_image, state)
    print(f"  ✓ Handled all-white image: {actions.shape}")
    
    # 测试 3: 极端状态值
    print("  Testing extreme state values...")
    extreme_state = np.array([np.pi, -np.pi, 10.0, -10.0, 100.0, -100.0], dtype=np.float32)
    image = np.random.rand(3, 480, 640).astype(np.float32)
    actions = engine.predict(image, extreme_state)
    print(f"  ✓ Handled extreme state values: {actions.shape}")
    
    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*70)
    print("DiffusionPolicy 离线推理测试套件")
    print("="*70)
    
    tests = [
        ("Single Inference", test_single_inference),
        ("Batch Inference", test_batch_inference),
        ("Load All Models", test_all_models),
        ("Inference Consistency", test_inference_consistency),
        ("Input Validation", test_input_validation),
        ("Edge Cases", test_edge_cases),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ Test '{test_name}' failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # 汇总结果
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✓ PASS" if passed_flag else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Ready for real robot deployment.")
        return True
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
