from typing import Protocol, Any

import numpy as np
import gymnasium as gym


class VecEnv(Protocol):
    observation_space: gym.spaces.Space
    action_space: gym.spaces.Space
    num_envs: int

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]: ...
    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]: ...
    def close(self) -> None: ...


from .vector_obs_normalizer import VecObsNormalizer, RunningMeanStd

__all__ = ["VecObsNormalizer", "RunningMeanStd", "VecEnv"]
