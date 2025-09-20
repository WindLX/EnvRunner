import numpy as np
import gymnasium as gym
import torch
from torch import nn


class FakeStochasticPolicy:
    """
    一个伪随机策略，用于为不同的向量化环境类型生成动作。
    """

    def __init__(
        self, action_space: gym.Space, is_vectorized_action_space: bool = False
    ):
        self.action_space = action_space
        self.is_vectorized = is_vectorized_action_space

    def get_actions(self, observations: np.ndarray) -> np.ndarray:
        """
        根据观察生成一批动作。

        Args:
            observations: 观察数组，形状为 (num_envs, ...)。

        Returns:
            动作数组，形状为 (num_envs, ...)。
        """
        num_envs = observations.shape[0]

        if self.is_vectorized:
            # 对于 gym 的 Sync/AsyncVectorEnv，动作空间已批处理
            return self.action_space.sample()
        else:
            # 对于我们的 EnvExecutor，动作空间是单个的
            return np.array([self.action_space.sample() for _ in range(num_envs)])


# --- 1. 定义一个模拟 GPU 策略 ---
class FakeGPUPolicy(nn.Module):
    """
    一个模拟在 GPU 上运行的神经网络策略。
    它包含一个简单的 MLP，并在 GPU 上执行前向传播。
    """

    def __init__(
        self, obs_dim: int, act_dim: int, action_space: gym.Space, device: torch.device
    ):
        super().__init__()
        self.device = device
        self.action_space = action_space

        # 创建一个简单的两层 MLP
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        ).to(device)

        # 确保 action space 是离散的，简化处理
        if not isinstance(self.action_space, gym.spaces.Discrete):
            raise TypeError("此伪策略目前仅支持 Discrete 动作空间用于演示。")

    @torch.no_grad()  # 关闭梯度计算以加速推理
    def get_actions(self, observations: np.ndarray) -> np.ndarray:
        """
        接收 NumPy 观测值，在 GPU 上推理，返回 NumPy 动作。
        """
        # 1. CPU -> GPU 数据传输
        obs_tensor = torch.from_numpy(observations).float().to(self.device)

        # 2. GPU 推理
        logits = self.net(obs_tensor)

        # 3. 从 logits 中采样动作 (e.g., argmax for deterministic)
        actions_tensor = torch.argmax(logits, dim=1)

        # 4. GPU -> CPU 数据传输
        actions_np = actions_tensor.cpu().numpy()

        return actions_np
