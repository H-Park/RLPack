from .lib import RLPack
import numpy as np
from typing import Dict, Any


class DQN:
    def __init__(self, model_name: str, model_args: Dict[str, Any], agent_args: Dict[str, Any]):
        self.get_dqn_agent = RLPack.GetDQNAgent(model_name, model_args, agent_args)

    def train(
            self,
            state_current: np.ndarray,
            state_next: np.ndarray,
            reward: float,
            action: int,
            done: bool
    ) -> int:
        return self.get_dqn_agent.train(
            state_current.astype(dtype=np.double),
            state_next.astype(dtype=np.double),
            reward,
            action,
            done,
            state_current.shape,
            state_next.shape
        )

    def policy(self, state_current: np.ndarray) -> int:
        return self.get_dqn_agent.policy(state_current, state_current.shape)
