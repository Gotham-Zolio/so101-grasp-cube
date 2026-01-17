import sys
print("[IMPORT] Starting imports...", flush=True)

import grasp_cube.real
print("[IMPORT] ✓ grasp_cube.real", flush=True)

import gymnasium as gym
print("[IMPORT] ✓ gymnasium", flush=True)

import tyro
print("[IMPORT] ✓ tyro", flush=True)

import dataclasses
print("[IMPORT] ✓ dataclasses", flush=True)

import matplotlib.pyplot as plt
print("[IMPORT] ✓ matplotlib", flush=True)

import numpy as np
print("[IMPORT] ✓ numpy", flush=True)

from grasp_cube.real.simple_fake_env import SimpleFakeEnvConfig, SimpleFakeEnv
print("[IMPORT] ✓ SimpleFakeEnv", flush=True)

from env_client import websocket_client_policy as _websocket_client_policy
print("[IMPORT] ✓ websocket_client_policy", flush=True)

from grasp_cube.real import MonitorWrapper, EvalRecordConfig, EvalRecordWrapper
print("[IMPORT] ✓ MonitorWrapper, EvalRecordConfig, EvalRecordWrapper", flush=True)

from collections import deque
print("[IMPORT] ✓ deque", flush=True)

print("[IMPORT] All imports successful!\n", flush=True)

@dataclasses.dataclass
class Args:
    env: SimpleFakeEnvConfig 
    eval: EvalRecordConfig
    host: str = "0.0.0.0"
    port: int = 8000
    monitor_host: str = "0.0.0.0"
    monitor_port: int = 9000
    num_episodes: int = 10
    
def main(args: Args):
    print("\n[CLIENT] Initializing...")
    
    # Create the simple fake environment
    print("[CLIENT] Creating SimpleFakeEnv...")
    env = SimpleFakeEnv(args.env)
    print(f"[CLIENT] ✓ SimpleFakeEnv created (task={args.env.task})")
    
    # ⚠️ TEMPORARILY SKIP WRAPPERS TO DEBUG
    print("[CLIENT] ⚠️  Skipping MonitorWrapper and EvalRecordWrapper for debugging")
    
    print(f"[CLIENT] Connecting to Server at ws://{args.host}:{args.port}...")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"[CLIENT] ✓ Connected to Server!")
    
    print(f"\n{'='*70}")
    print(f"DiffusionPolicy Client Started (SIMPLIFIED)")
    print(f"{'='*70}")
    print(f"✓ Server: ws://{args.host}:{args.port}")
    print(f"✓ Episodes: {args.num_episodes}")
    print(f"⚠️  MonitorWrapper and EvalRecordWrapper disabled for debugging")
    print(f"{'='*70}\n")
    
    for episode in range(args.num_episodes):
        print(f"\n🎬 Episode {episode+1}/{args.num_episodes}")
        print(f"[RESET] Calling env.reset()...", flush=True)
        obs, info = env.reset()
        print(f"[RESET] ✓ env.reset() completed", flush=True)
        
        print(f"[RESET] Calling client.reset()...", flush=True)
        client.reset()
        print(f"[RESET] ✓ client.reset() completed", flush=True)
        done = False
        action_plan = deque()
        actions = []
        gt_actions = []
        step_count = 0
        
        while not done:
            if not action_plan:
                # Get action from server
                response = client.infer(obs)
                print(f"  [DEBUG] Response type: {type(response)}")
                
                # Response should be a list of lists: [[action1], [action2], ...]
                action_list = response
                print(f"  [DEBUG] action_list type: {type(action_list)}, len: {len(action_list) if hasattr(action_list, '__len__') else 'N/A'}")
                
                action_chunk = np.array(action_list)  # Convert list to numpy array
                print(f"  [DEBUG] action_chunk shape: {action_chunk.shape}, ndim: {action_chunk.ndim}")
                
                if action_chunk.ndim == 0:
                    print(f"  [ERROR] Got 0-d array! Value: {action_chunk}")
                    raise RuntimeError(f"Got 0-d array from server")
                
                action_plan.extend(action_chunk)
                print(f"  ✓ Received action sequence with shape {action_chunk.shape}")
            
            action = action_plan.popleft()
            obs, reward, done, truncated, info = env.step(action)
            actions.append(action)
            gt_action = info.get("action")
            assert gt_action is not None, "Ground truth action missing in info"
            gt_actions.append(gt_action)
            step_count += 1
            
            if step_count % 10 == 0:
                print(f"  Step {step_count}: action shape={action.shape}, done={done}")
            
        
        actions = np.array(actions)
        gt_actions = np.array(gt_actions)
        steps = np.arange(len(actions))
        num_actions = actions.shape[1]
        # draw 1 x num_actions subplots
        fig, axs = plt.subplots(num_actions, 1, figsize=(8, 4 * num_actions))
        for i in range(num_actions):
            axs[i].plot(steps, actions[:, i], label="Predicted Action")
            if gt_actions[:, i].any():
                axs[i].plot(steps, gt_actions[:, i], label="Ground Truth Action")
            axs[i].set_xlabel("Step")
            axs[i].set_ylabel(f"Action {i}")
            axs[i].legend()
            axs[i].grid()
        plt.tight_layout()
        plt.savefig(env.run_dir / f"episode_{episode}_actions.png")
        plt.close(fig)
        print(f"  ✓ Episode {episode+1} completed ({step_count} steps)")
    
    print(f"\n{'='*70}")
    print(f"✅ All episodes completed!")
    print(f"Results saved to: {env.run_dir}")
    print(f"MonitorWrapper: http://127.0.0.1:{args.monitor_port}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    print("[MAIN] Parsing arguments...", flush=True)
    args = tyro.cli(Args)
    print(f"[MAIN] Arguments parsed: task={args.env.task}, episodes={args.num_episodes}", flush=True)
    print("[MAIN] Calling main()...", flush=True)
    main(args)
    print("[MAIN] main() completed!", flush=True)
