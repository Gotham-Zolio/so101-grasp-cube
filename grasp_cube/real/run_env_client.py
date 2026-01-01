import grasp_cube.real
import gymnasium as gym
import tyro
import dataclasses
import matplotlib.pyplot as plt
import numpy as np
from grasp_cube.real.lerobot_env import LeRobotEnvConfig
from env_client import websocket_client_policy as _websocket_client_policy
from grasp_cube.real import MonitorWrapper, EvalRecordConfig, EvalRecordWrapper
from collections import deque

@dataclasses.dataclass
class Args:
    env: LeRobotEnvConfig 
    eval: EvalRecordConfig
    host: str = "0.0.0.0"
    port: int = 8000
    monitor_host: str = "0.0.0.0"
    monitor_port: int = 9000
    num_episodes: int = 10
    
def main(args: Args):
    env = gym.make(
        "LeRobotEnv-v0",
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
        while not done:
            if not action_plan:
                action_chunk = client.infer(obs)["action"]
                action_plan.extend(action_chunk)
            action = action_plan.popleft()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
