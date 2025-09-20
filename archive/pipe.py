# benchmark_final.py

import gymnasium as gym
import numpy as np
import time
import multiprocessing as mp
import pandas as pd
import torch
import torch.nn as nn
from typing import Callable

# --- 导入所有必要的模块 ---
# TODO: 步骤 1 - 确保所有自定义模块都可以被导入
from envrunner import EnvExecutor as HighPerformanceEnvExecutor  # 假设您已将其打包
from envrunner import PipelinedExecutor
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

from pygtm_env.task.upset_recovery import EnvBuilder
from conflga import conflga_func, ConflgaConfig

# 为了脚本可运行，先用占位符
IS_CUSTOM_ENV = True
# class EnvBuilder: pass
# def get_conflga_config(): return None


# --- 1. 可序列化的环境构造器 (与之前相同) ---
class EnvMaker:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def __call__(self) -> gym.Env:
        if IS_CUSTOM_ENV:
            return EnvBuilder(self.cfg)()
        else:
            return gym.make("CartPole-v1")


# --- 2. 模拟 GPU 策略 (与之前相同，但修复了动作空间问题) ---
class FakeGPUPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_space: gym.Space, device: torch.device):
        super().__init__()
        self.device = device
        self.action_space = act_space

        # 确定 act_dim
        if isinstance(act_space, gym.spaces.Discrete):
            act_dim = act_space.n
        elif isinstance(act_space, gym.spaces.Box):
            act_dim = act_space.shape[0]
        else:
            raise TypeError(f"Unsupported action space {type(act_space)}")

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim),
        ).to(device)

    @torch.no_grad()
    def get_actions(self, observations: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(observations).float().to(self.device)
        if isinstance(self.action_space, gym.spaces.Discrete):
            logits = self.net(obs_tensor)
            actions_tensor = torch.argmax(logits, dim=1)
            return actions_tensor.cpu().numpy()
        elif isinstance(self.action_space, gym.spaces.Box):
            # 对于连续动作，通常是输出均值
            actions_tensor = self.net(obs_tensor)
            return actions_tensor.cpu().numpy()


# --- 3. 核心测量函数 ---
@conflga_func(config_dir="conf", default_config="gtm_env", auto_print=False)
def run_trial(
    cfg: ConflgaConfig,
    algorithm_name: str,
    algorithm_cls: Callable,
    num_envs: int,
    num_workers: int,
    effective_batch_size: int,
    policy: FakeGPUPolicy,
    num_steps: int = 100,  # 运行固定的步数/批次数
) -> float:

    # cfg = get_conflga_config() if IS_CUSTOM_ENV else None
    # maker = EnvMaker(cfg)
    # env_fns = [maker for _ in range(num_envs)]
    # 使用占位符
    maker = EnvMaker(cfg)
    env_fns = [maker for _ in range(num_envs)]

    env = None
    total_frames = 0
    total_time = 0

    try:
        # --- 流水线执行器的特殊处理 ---
        if algorithm_name == "PipelinedExecutor":
            env = PipelinedExecutor(env_fns, num_workers, effective_batch_size, policy)

            # 预热
            warmup_steps = 5
            for i, batch in enumerate(env):
                if i >= warmup_steps:
                    break
                _ = batch["obs"].to(policy.device, non_blocking=True)

            # 测量
            start_time = time.perf_counter()
            for i, batch in enumerate(env):
                if i >= num_steps:
                    break
                # 模拟训练/使用数据
                _ = batch["obs"].to(policy.device, non_blocking=True)
                # time.sleep(0.001) # 可以模拟一个小的训练延迟
            end_time = time.perf_counter()

            total_time = end_time - start_time
            total_frames = num_steps * effective_batch_size
            return total_frames / total_time

        # --- 非流水线执行器的处理 ---
        else:
            if algorithm_cls is HighPerformanceEnvExecutor:
                env = algorithm_cls(env_fns, num_workers=num_workers)
                is_vectorized_action_space = False
            else:
                env = algorithm_cls(env_fns)
                is_vectorized_action_space = True

            # 对于非流水线，策略的动作空间需要单独处理
            local_policy = FakeGPUPolicy(
                policy.net[0].in_features,  # obs_dim
                env.action_space if is_vectorized_action_space else policy.action_space,
                policy.device,
            )

            obs, _ = env.reset(seed=42)

            # 预热
            for _ in range(5):
                actions = local_policy.get_actions(obs)
                obs, _, _, _, _ = env.step(actions)

            # 测量
            start_time = time.perf_counter()
            for _ in range(num_steps):
                actions = local_policy.get_actions(obs)
                obs, _, _, _, _ = env.step(actions)
            end_time = time.perf_counter()

            total_time = end_time - start_time
            total_frames = num_steps * effective_batch_size
            return total_frames / total_time

    except Exception as e:
        # print(f"\n  - WARN: Config failed with error: {type(e).__name__}: {e}")
        return 0.0
    finally:
        if env:
            env.close()


# --- 4. 主搜索程序 ---
def main():
    # --- 搜索空间 ---
    BATCH_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]
    WORKER_COUNTS = [4, 8, 16, 24, 32]
    ALGORITHMS = {
        # 新增 PipelinedExecutor
        "PipelinedExecutor": PipelinedExecutor,
        "EnvExecutor": HighPerformanceEnvExecutor,
        "AsyncVectorEnv": AsyncVectorEnv,
        "SyncVectorEnv": SyncVectorEnv,
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("      最终全系统性能基准测试 (含流水线)")
    # ... (打印硬件和目标信息)

    # --- 初始化共享策略 ---
    # TODO: 步骤 3 - 如果是自定义环境，确保这里的维度是正确的
    obs_dim, act_space = (
        (4, gym.spaces.Discrete(2))
        if not IS_CUSTOM_ENV
        else (17, gym.spaces.Discrete(4))
    )  # 占位符
    shared_policy = FakeGPUPolicy(obs_dim, act_space, device)

    # --- 创建测试配置 ---
    all_configs = []
    for algo_name, algo_cls in ALGORITHMS.items():
        if algo_name in ["EnvExecutor", "PipelinedExecutor"]:
            for batch_size in BATCH_SIZES:
                for workers in WORKER_COUNTS:
                    if batch_size % workers == 0:
                        all_configs.append((algo_name, algo_cls, batch_size, workers))
        elif algo_name == "AsyncVectorEnv":
            for workers in WORKER_COUNTS:
                all_configs.append((algo_name, algo_cls, workers, workers))
        elif algo_name == "SyncVectorEnv":
            for batch_size in BATCH_SIZES:
                all_configs.append((algo_name, algo_cls, batch_size, 1))

    # --- 运行搜索 ---
    results = []
    total_configs = len(all_configs)
    print(f"[*] 将要测试 {total_configs} 种不同配置...\n")

    for i, (algo_name, algo_cls, batch_size, workers) in enumerate(all_configs):
        effective_batch_size = workers if algo_name == "AsyncVectorEnv" else batch_size
        num_envs_to_create = workers if algo_name == "AsyncVectorEnv" else batch_size

        print(
            f"[{i+1}/{total_configs}] 测试: {algo_name:<18} Batch={batch_size:<5} Workers={workers:<3}... ",
            end="",
            flush=True,
        )

        fps = run_trial(
            algo_name,
            algo_cls,
            num_envs_to_create,
            workers,
            effective_batch_size,
            shared_policy,
        )

        print(f"-> FPS: {fps:,.0f}")
        results.append([algo_name, batch_size, workers, fps])

    # --- 报告结果 (与之前脚本完全相同) ---
    print("\n\n" + "=" * 70)
    print("      搜索结果报告")
    df = pd.DataFrame(
        results, columns=["Algorithm", "Batch Size", "Workers", "Sync FPS"]
    )
    df = df.sort_values(by="Sync FPS", ascending=False).reset_index(drop=True)
    df["Sync FPS"] = df["Sync FPS"].map("{:,.0f}".format)
    print(df.to_string())
    print("\n" + "-" * 70)
    best_config = df.iloc[0]
    print("\n🏆 最优配置建议 🏆")
    print(f"\n  - 算法:         {best_config['Algorithm']}")
    # ... (打印最优配置和分析)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
