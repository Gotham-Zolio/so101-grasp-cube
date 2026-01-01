import grasp_cube.real
import gymnasium as gym
import tyro
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
from grasp_cube.real.fake_lerobot_env import FakeLeRobotEnvConfig
from env_client import websocket_client_policy as _websocket_client_policy
from grasp_cube.real import MonitorWrapper, EvalRecordConfig, EvalRecordWrapper
from collections import deque

@dataclasses.dataclass
class Args:
    env: FakeLeRobotEnvConfig 
    eval: EvalRecordConfig
    host: str = "0.0.0.0"
    port: int = 8000
    monitor_host: str = "0.0.0.0"
    monitor_port: int = 9000
    num_episodes: int = 10
    
def main(args: Args):
    env = gym.make(
        "FakeLeRobotEnv-v0",
        config=args.env,
    )
    env = MonitorWrapper(env, port=args.monitor_port, host=args.monitor_host, include_images=True)
    env = EvalRecordWrapper(env, config=args.eval)
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        client.reset()
        done = False
        action_plan = deque()
        actions = []
        gt_actions = []
        while not done:
            if not action_plan:
                action_chunk = client.infer(obs)["action"]
                action_plan.extend(action_chunk)
            action = action_plan.popleft()
            obs, reward, done, truncated, info = env.step(action)
            actions.append(action)
            gt_action = info.get("gt_action")
            assert gt_action is not None, "Ground truth action missing in info"
            gt_actions.append(gt_action)
            
        
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
        
if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
