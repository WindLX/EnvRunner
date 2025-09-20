import time
import multiprocessing as mp
from typing import Callable

import numpy as np
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

from envrunner import EnvExecutor
from cartpole import CartPoleEnv
from utils import FakeStochasticPolicy


# --- 基准测试运行函数 ---
def measure_batch_collection_time(
    vector_env_cls: Callable,
    num_envs: int,
    num_workers: int,
    target_batch_size: int,
    num_trials: int = 100,
) -> float:
    """
    测量收集一个特定大小批处理数据所需的平均时间。

    Args:
        vector_env_cls: 要测试的向量化环境类。
        num_envs: 传递给环境构造函数的环境数量。
        num_workers: 工作进程数。
        target_batch_size: 目标收集的数据帧数量。
        num_trials: 重复测试的次数以取平均值。

    Returns:
        收集单批数据所需的平均时间（毫秒）。
    """

    env_fns = [lambda: CartPoleEnv() for _ in range(num_envs)]
    env = None

    try:
        if vector_env_cls is EnvExecutor:
            env = vector_env_cls(env_fns, num_workers=num_workers)
            policy = FakeStochasticPolicy(
                env.action_space, is_vectorized_action_space=False
            )
        else:  # For Sync and Async
            env = vector_env_cls(env_fns)
            policy = FakeStochasticPolicy(
                env.action_space, is_vectorized_action_space=True
            )

        dummy_obs = np.zeros(
            (num_envs,) + env.single_observation_space.shape,
            dtype=env.single_observation_space.dtype,
        )
        env.reset(seed=42)

        # 预热一次，以排除首次调用的额外开销
        env.step(policy.get_actions(dummy_obs))

        # 计算需要循环多少次 step 才能达到目标 batch size
        steps_per_batch = target_batch_size // num_envs

        total_time = 0
        for _ in range(num_trials):
            start_time = time.perf_counter()
            for _ in range(steps_per_batch):
                actions = policy.get_actions(dummy_obs)
                env.step(actions)
            end_time = time.perf_counter()
            total_time += end_time - start_time

        avg_time_ms = (total_time / num_trials) * 1000
        return avg_time_ms

    finally:
        if env:
            env.close()


# --- 主函数 ---
def main():
    TARGET_BATCH_SIZE = 4096  # 目标批处理大小（帧）
    NUM_WORKERS = 32

    assert (
        TARGET_BATCH_SIZE % NUM_WORKERS == 0
    ), "TARGET_BATCH_SIZE 必须能被 NUM_WORKERS 整除。"

    CPU_CORES = mp.cpu_count()

    print("=" * 60)
    print("      单批数据收集性能基准测试")
    print(f"目标批处理大小: {TARGET_BATCH_SIZE} 帧")
    print(f"本机 CPU 核心数: {CPU_CORES}")
    print(f"EnvExecutor 使用进程数: {NUM_WORKERS}")
    print("=" * 60)

    # --- 1. 测试 HighPerformanceEnvExecutor ---
    print(
        f"[*] 正在测试 EnvExecutor ({TARGET_BATCH_SIZE} envs, {NUM_WORKERS} workers), {TARGET_BATCH_SIZE // NUM_WORKERS} envs per worker...",
        flush=True,
    )
    time_hpe = measure_batch_collection_time(
        EnvExecutor,
        num_envs=TARGET_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        target_batch_size=TARGET_BATCH_SIZE,
    )
    print(f"    -> 平均耗时: {time_hpe:.2f} ms\n")

    # --- 2. 测试 AsyncVectorEnv ---
    # AsyncVectorEnv 的进程数等于环境数，我们将其设置为与 EnvExecutor 相同
    async_envs = NUM_WORKERS
    print(
        f"[*] 正在测试 AsyncVectorEnv ({async_envs} envs, {async_envs} workers), 1 env per worker...",
        flush=True,
    )
    time_async = measure_batch_collection_time(
        AsyncVectorEnv,
        num_envs=async_envs,
        num_workers=async_envs,  # 此参数在函数内部被忽略，仅为占位
        target_batch_size=TARGET_BATCH_SIZE,
    )
    print(f"    -> 平均耗时: {time_async:.2f} ms\n")

    # --- 3. 测试 SyncVectorEnv ---
    print(
        f"[*] 正在测试 SyncVectorEnv ({TARGET_BATCH_SIZE} envs, 1 worker, {TARGET_BATCH_SIZE} envs per worker)...",
        flush=True,
    )
    time_sync = measure_batch_collection_time(
        SyncVectorEnv,
        num_envs=TARGET_BATCH_SIZE,
        num_workers=1,  # 占位
        target_batch_size=TARGET_BATCH_SIZE,
    )
    print(f"    -> 平均耗时: {time_sync:.2f} ms\n")

    # --- 结果汇总 ---
    print("-" * 60)
    print(f"结果汇总 (收集 {TARGET_BATCH_SIZE} 帧数据所需时间):")
    print(f"  - EnvExecutor:      {time_hpe:7.2f} ms")
    print(f"  - AsyncVectorEnv:   {time_async:7.2f} ms")
    print(f"  - SyncVectorEnv:    {time_sync:7.2f} ms")
    print("-" * 60)


if __name__ == "__main__":
    main()
