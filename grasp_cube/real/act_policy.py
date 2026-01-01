import dataclasses
import tyro
import pathlib
import numpy as np
import torch
from typing import Any, Literal
from lerobot.policies.act.configuration_act import ACTConfig, PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.utils import prepare_observation_for_inference
from lerobot.policies.factory import make_pre_post_processors

@dataclasses.dataclass
class LeRobotACTPolicyConfig:
    path: pathlib.Path
    robot_type: Literal["so101", "bi_so101"] = "so101"
    device: str = "cuda"
    act_steps: int | None = None
    
class LeRobotACTPolicy:
    def __init__(self, config: LeRobotACTPolicyConfig):
        policy_config = PreTrainedConfig.from_pretrained(config.path)
        assert isinstance(policy_config, ACTConfig), f"Expected ACTConfig, got {type(policy_config)}"
        policy = ACTPolicy.from_pretrained(config.path, config=policy_config)
        self.policy = policy
        self.robot_type = config.robot_type
        self.device = torch.device(config.device)
        self.policy.to(self.device)
        self.policy.eval()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=str(config.path),
            preprocessor_overrides={
                "device_processor": {"device": config.device},
            },
        )
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.act_steps = config.act_steps

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        obs = {}
        if self.robot_type == "so101":
            obs["observation.state"] = observation["states"]["arm"]
            obs["observation.images.front"] = observation["images"]["front"]
            obs["observation.images.wrist"] = observation["images"]["wrist"]
        elif self.robot_type == "bi_so101":
            obs["observation.state"] = np.concatenate([
                observation["states"]["left_arm"],
                observation["states"]["right_arm"],
            ], axis=-1)
            obs["observation.images.front"] = observation["images"]["front"]
            obs["observation.images.left_wrist"] = observation["images"]["left_wrist"]
            obs["observation.images.right_wrist"] = observation["images"]["right_wrist"]
        obs_infer = prepare_observation_for_inference(obs, self.device, observation["task"], self.robot_type)
        obs_infer_processed = self.preprocessor(obs_infer)
        action_chunk = self.policy.predict_action_chunk(obs_infer_processed).swapaxes(0, 1).cpu()
        for i in range(len(action_chunk)):
            action_chunk[i] = self.postprocessor(action_chunk[i])
        if self.act_steps is None:
            return action_chunk[:, 0].numpy()
        else:
            return action_chunk[:self.act_steps, 0].numpy()
    
    def reset(self):
        print("Resetting LeRobotACTPolicy")
        self.policy.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()
    
if __name__ == "__main__":
    config = tyro.cli(LeRobotACTPolicyConfig)
    print(config)
    policy = LeRobotACTPolicy(config)
    print("ACT Policy initialized.")