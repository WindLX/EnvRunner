from typing import Callable
import time
import multiprocessing as mp

import numpy as np
import torch
import polars as pl
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

from envrunner import EnvExecutor
from cartpole import CartPoleEnv
from utils import FakeStochasticPolicy, FakeGPUPolicy


class EnvMaker:
    """
    一个可序列化的类，用于创建环境实例。
    """

    def __init__(self):
        pass

    def __call__(self) -> gym.Env:
        """
        这个方法将在子进程中被调用，以创建一个新的环境实例。
        """
        return CartPoleEnv()


# --- 2. 核心测量函数 (更新后) ---
def run_trial_with_gpu(
    algorithm_cls: Callable,
    num_envs: int,
    num_workers: int,
    effective_batch_size: int,
    policy: FakeGPUPolicy,
    num_trials: int = 50,
) -> float:
    """运行一次包含 GPU 推理的测试并返回 FPS。"""

    # 步骤 2 - 配置您的环境构造函数
    # ==============================================================================

    env_fns = [EnvMaker() for _ in range(num_envs)]
    # ==============================================================================

    env = None
    try:
        # 初始化环境执行器
        if algorithm_cls is EnvExecutor:
            env = algorithm_cls(env_fns, num_workers=num_workers)
        else:
            env = algorithm_cls(env_fns)

        # 使用传入的 policy
        obs, _ = env.reset(seed=42)

        # 预热 (包括JIT编译和GPU缓存)
        for _ in range(5):
            actions = policy.get_actions(obs)
            obs, _, _, _, _ = env.step(actions)

        total_time = 0
        for _ in range(num_trials):
            start = time.perf_counter()
            # 注意：这里我们使用真实的 obs
            actions = policy.get_actions(obs)
            obs, _, _, _, _ = env.step(actions)
            end = time.perf_counter()
            total_time += end - start

        avg_time_per_step = total_time / num_trials
        fps = effective_batch_size / avg_time_per_step
        return fps

    except Exception as e:
        print(f"  - WARN: Config failed with error: {e}")
        return 0.0
    finally:
        if env:
            env.close()


# --- 核心测量函数 ---
def run_trial(
    algorithm_cls: Callable,
    num_envs: int,
    num_workers: int,
    effective_batch_size: int,
    num_trials: int = 20,  # 减少 trial 次数以加快搜索
) -> float:
    """运行一次测试并返回 FPS。"""

    env_fns = [lambda: CartPoleEnv() for _ in range(num_envs)]
    # ==============================================================================

    env = None
    try:
        if algorithm_cls is EnvExecutor:
            env = algorithm_cls(env_fns, num_workers=num_workers)
            policy = FakeStochasticPolicy(
                env.action_space, is_vectorized_action_space=False
            )
        else:
            env = algorithm_cls(env_fns)
            policy = FakeStochasticPolicy(
                env.action_space, is_vectorized_action_space=True
            )

        dummy_obs = np.zeros(
            (num_envs,) + env.single_observation_space.shape,
            dtype=env.single_observation_space.dtype,
        )
        env.reset(seed=42)

        # 预热
        env.step(policy.get_actions(dummy_obs))

        total_time = 0
        for _ in range(num_trials):
            start = time.perf_counter()
            env.step(policy.get_actions(dummy_obs))
            end = time.perf_counter()
            total_time += end - start

        avg_time_per_step = total_time / num_trials
        fps = effective_batch_size / avg_time_per_step
        return fps

    except Exception as e:
        # 捕获错误并返回 0 FPS，表示此配置不可行
        # print(f"  - WARN: Config failed with error: {e}")
        return 0.0
    finally:
        if env:
            env.close()


# --- 主搜索程序 ---
def main():
    # --- 1. 定义搜索空间 ---
    # 您的 GPU 内存巨大，可以承受非常大的批次
    BATCH_SIZES = [256, 512, 1024, 2048, 4096, 8192, 16384]
    # i9-14900K: 8 P-cores, 16 E-cores. 测试 P-core 数量，总核心数，总线程数等
    WORKER_COUNTS = [4, 8, 12, 16, 24, 32]
    ALGORITHMS = {
        "EnvExecutor": EnvExecutor,
        "AsyncVectorEnv": AsyncVectorEnv,
        "SyncVectorEnv": SyncVectorEnv,
    }

    cpu_cores = mp.cpu_count()
    # --- GPU 设置 ---
    if not torch.cuda.is_available():
        print("错误: 本脚本需要 CUDA-enabled GPU。")
        return
    # 优先使用第一块 RTX 5090
    device = torch.device("cuda:0")
    print("=" * 70)
    print("      全系统 (CPU+GPU) RL 采样配置搜索程序")
    print(f"硬件: i9-14900K ({cpu_cores} 核心), 2x RTX 5090 (使用 {device})")
    print(f"目标: 最大化端到端的同步 FPS")
    print("=" * 70)

    # --- 初始化共享的 GPU 策略 ---
    # 一个临时环境来推断 obs 和 action 的维度
    temp_env = CartPoleEnv()
    obs_dim = temp_env.observation_space.shape[0]  # type: ignore
    act_dim = temp_env.action_space.n  # type: ignore
    temp_env.close()

    # 创建一个将在所有测试中共享的策略实例
    shared_policy = FakeGPUPolicy(
        obs_dim, act_dim, gym.spaces.Discrete(act_dim), device
    )

    # --- 2. 创建所有测试配置 ---
    all_configs = []
    for algo_name, algo_cls in ALGORITHMS.items():
        if algo_name == "EnvExecutor":
            for batch_size in BATCH_SIZES:
                for workers in WORKER_COUNTS:
                    if batch_size % workers == 0:  # 确保可以整除
                        all_configs.append((algo_name, algo_cls, batch_size, workers))
        elif algo_name == "AsyncVectorEnv":
            # 对于 Async, 它的同步批大小就是其进程数
            for workers in WORKER_COUNTS:
                all_configs.append(
                    (algo_name, algo_cls, workers, workers)
                )  # batch_size = workers
        elif algo_name == "SyncVectorEnv":
            # 对于 Sync, 它只有一个进程
            for batch_size in BATCH_SIZES:
                all_configs.append((algo_name, algo_cls, batch_size, 1))

    # --- 3. 运行搜索 ---
    results = []
    total_configs = len(all_configs)
    print(f"[*] 将要测试 {total_configs} 种不同配置...\n")

    for i, (algo_name, algo_cls, batch_size, workers) in enumerate(all_configs):

        # 对于 Async, effective_batch_size 就是 workers
        effective_batch_size = workers if algo_name == "AsyncVectorEnv" else batch_size
        num_envs_to_create = workers if algo_name == "AsyncVectorEnv" else batch_size

        print(
            f"[{i+1}/{total_configs}] 测试: {algo_name}, Batch={batch_size}, Workers={workers}... ",
            end="",
            flush=True,
        )

        # fps = run_trial(algo_cls, num_envs_to_create, workers, effective_batch_size)
        fps = run_trial_with_gpu(
            algo_cls, num_envs_to_create, workers, effective_batch_size, shared_policy
        )

        print(f"-> FPS: {fps:,.0f}")
        results.append([algo_name, batch_size, workers, fps])

    # --- 4. 报告结果 ---
    print("\n\n" + "=" * 70)
    print("      搜索结果报告")
    print("=" * 70)

    # 使用 polars 创建和格式化表格
    df = pl.DataFrame(
        results, schema=["Algorithm", "Batch Size", "Workers", "Sync FPS"], orient="row"
    )
    df = df.sort("Sync FPS", descending=True)

    # 将 FPS 格式化为带逗号的整数
    df = df.with_columns(
        pl.col("Sync FPS").map_elements(lambda x: "{:,.0f}".format(x)).alias("Sync FPS")
    )

    print(str(df))

    print("\n" + "-" * 70)
    best_config = df.row(0)
    print("\n🏆 最优配置建议 🏆")
    print(f"\n  - 算法:         {best_config[0]}")
    print(f"  - 批处理大小:   {best_config[1]}")
    print(f"  - 工作进程数:   {best_config[2]}")
    print(f"  - 预估同步FPS:  {best_config[3]}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
