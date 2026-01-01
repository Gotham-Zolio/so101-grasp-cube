import abc
import numpy as np
from typing import Dict


class BasePolicy(abc.ABC):
    def get_action(self, obs: Dict) -> np.ndarray:
        """Infer actions from observations."""
        ...

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass
