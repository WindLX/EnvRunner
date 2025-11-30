from typing import Protocol, Any
from enum import Enum

import numpy as np
import gymnasium as gym


class VecEnv(Protocol):
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    num_envs: int

    def reset(self) -> tuple[np.ndarray, dict[str, Any]]: ...
    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]: ...
    def close(self) -> None: ...


class AutoResetMode(Enum):
    DISABLE = "disable"
    NEXT_STEP = "next_step"
