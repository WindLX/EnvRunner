import gymnasium as gym
import numpy as np
import time
import multiprocessing as mp
from torch import nn
import torch
import pandas as pd  # 使用 pandas 来美化输出结果
from typing import Callable

from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from pygtm_env.task.upset_recovery import EnvBuilder
from conflga import conflga_func, ConflgaConfig

from envrunner import EnvExecutor, PipelinedExecutor

IS_CUSTOM_ENV = True  # 切换为 False 以使用占位符环境


class EnvMaker:
    """
    一个可序列化的类，用于创建环境实例。
    """

    def __init__(self, cfg):
        # 将配置存储为实例属性
        self.cfg = cfg
        # 如果有其他需要的参数，也可以在这里传入

    def __call__(self) -> gym.Env:
        """
        这个方法将在子进程中被调用，以创建一个新的环境实例。
        """
        # TODO: 将您原来的 make_env 或 lambda 的逻辑放在这里
        if IS_CUSTOM_ENV:
            return EnvBuilder(self.cfg)()
        else:
            return gym.make("CartPole-v1")


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


@conflga_func(config_dir="conf", default_config="gtm_env", auto_print=False)
def train(cfg: ConflgaConfig):
    # --- 1. 初始化 ---
    BATCH_SIZE = 8192
    NUM_WORKERS = 32

    # ... 创建 env_fns, policy ...
    maker = EnvMaker(cfg)
    env_fns = [maker for _ in range(BATCH_SIZE)]

    # 创建一个将在所有测试中共享的策略实例
    policy = FakeGPUPolicy(17, 4, gym.spaces.Discrete(4), device=torch.device("cuda:0"))

    # 初始化流水线执行器
    data_sampler = PipelinedExecutor(env_fns, NUM_WORKERS, BATCH_SIZE, policy)

    # --- 2. 训练循环 ---
    # `for batch in data_sampler:` 这个循环会自动处理所有等待和同步
    # 当你的 GPU 正在处理一个 batch 时，CPU 已经在后台准备下一个 batch 了
    for step, batch in enumerate(data_sampler):
        if step >= 100:  # 假设只训练 100 步
            break

        start_train_time = time.perf_counter()

        # batch['obs'] 已经是 GPU 张量了
        obs_on_gpu = batch["obs"]

        # ... 在这里执行您的训练逻辑 (e.g., PPO update) ...
        # model.train(obs_on_gpu, ...)
        time.sleep(0.01)  # 模拟训练耗时

        end_train_time = time.perf_counter()
        print(
            f"Step {step}: Training on batch took {(end_train_time - start_train_time)*1000:.2f} ms"
        )

    # --- 3. 清理 ---
    data_sampler.close()


if __name__ == "__main__":
    train()
