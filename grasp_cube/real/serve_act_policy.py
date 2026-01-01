from env_client import websocket_policy_server as _websocket_policy_server
from grasp_cube.real.act_policy import LeRobotACTPolicyConfig, LeRobotACTPolicy
import dataclasses
import tyro

@dataclasses.dataclass
class ActPolicyServerConfig:
    policy: LeRobotACTPolicyConfig
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    
def create_act_policy_server(config: ActPolicyServerConfig) -> _websocket_policy_server.WebsocketPolicyServer:
    policy = LeRobotACTPolicy(config.policy)
    server = _websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=config.host,
        port=config.port,
        metadata={"policy_type": "ACTPolicy"},
    )
    return server

if __name__ == "__main__":
    config = tyro.cli(ActPolicyServerConfig)
    server = create_act_policy_server(config)
    server.serve_forever()