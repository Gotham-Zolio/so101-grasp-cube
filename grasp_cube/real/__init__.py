import gymnasium as gym

from .monitor_wrapper import MonitorWrapper
from .eval_record_wrapper import EvalRecordWrapper, EvalRecordConfig

try:
    from .fake_lerobot_env import FakeLeRobotEnv
    gym.register("FakeLeRobotEnv-v0", entry_point="grasp_cube.real.fake_lerobot_env:FakeLeRobotEnv")
except Exception as e:
    print(f"Warning: could not import FakeLeRobotEnv due to error: {e}")

try:
    from .lerobot_env import LeRobotEnv
    gym.register("LeRobotEnv-v0", entry_point="grasp_cube.real.lerobot_env:LeRobotEnv")
except Exception as e:
    print(f"Warning: could not import LeRobotEnv due to error: {e}")
