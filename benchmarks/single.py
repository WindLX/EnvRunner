import time
import multiprocessing as mp
from typing import Callable

from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
import numpy as np

from envrunner import EnvExecutor
from cartpole import CartPoleEnv
from utils import FakeStochasticPolicy


def run_benchmark(
    vector_env_cls: Callable,
    env_fns: list[Callable],
    num_workers: int,  # 这个参数对 AsyncVectorEnv 和 SyncVectorEnv 无效，但为了接口统一我们保留
    total_steps: int,
) -> float:
    """
    为给定的向量化环境类运行一个标准的采样循环并返回 FPS。
    """
    env = None
    try:
        if vector_env_cls is EnvExecutor:
            env = vector_env_cls(env_fns, num_workers=num_workers)
        elif vector_env_cls is AsyncVectorEnv:
            # AsyncVectorEnv 为每个 env_fn 创建一个进程，它没有 num_workers 参数
            env = vector_env_cls(env_fns)
        elif vector_env_cls is SyncVectorEnv:
            # SyncVectorEnv 在主进程中运行，也没有 num_workers 参数
            env = vector_env_cls(env_fns)
        else:
            raise TypeError(f"不支持的向量化环境类型: {vector_env_cls}")

        policy = FakeStochasticPolicy(
            env.action_space, is_vectorized_action_space=False
        )
        if vector_env_cls is not EnvExecutor:
            policy.is_vectorized = True

        start_time = time.perf_counter()

        # 对于 AsyncVectorEnv，它的 num_envs 就是进程数
        # 为了公平比较，我们应该让它运行相同的总步数
        num_envs_for_step = env.num_envs

        # 修正：obs 的形状应该和环境数匹配
        # 在 `get_actions` 中我们已经处理了 num_envs，所以这里用 0 填充即可
        dummy_obs = np.zeros(
            (num_envs_for_step,) + env.single_observation_space.shape,
            dtype=env.single_observation_space.dtype,
        )

        env.reset(seed=42)
        # 修正：循环次数应该是 total_steps / num_envs
        for _ in range(total_steps // num_envs_for_step):
            actions = policy.get_actions(dummy_obs)
            env.step(actions)

        end_time = time.perf_counter()

        # 修正：计算 FPS 时，总步数应该是循环次数 * 环境数
        actual_steps = (total_steps // num_envs_for_step) * num_envs_for_step
        total_time = end_time - start_time
        fps = actual_steps / total_time
        return fps

    finally:
        if env:
            env.close()


# --- 主函数 ---
def main():
    TOTAL_STEPS = 50_000  # 每个基准测试运行的总步数

    # --- 定义测试配置 ---
    # (环境总数, 工作进程数)
    BENCHMARK_CONFIGS = [
        (8, 4),
        (16, 4),
        (32, 4),
        (64, 4),
        (16, 8),
        (32, 8),
        (64, 8),
    ]

    # 获取当前机器的 CPU 核心数作为参考
    cpu_cores = mp.cpu_count()
    print("=" * 60)
    print("      环境执行器性能基准测试")
    print(f"环境: CartPole (custom)")
    print(f"CPU 核心数: {cpu_cores}")
    print(f"每个测试的总步数: {TOTAL_STEPS}")
    print("=" * 60)

    # 打印表头
    print(
        f"{'Config (Envs, Workers)':<25} | {'EnvExecutor FPS':>18} | {'AsyncVectorEnv FPS':>20} | {'SyncVectorEnv FPS':>18}"
    )
    print("-" * 88)

    for num_envs, num_workers in BENCHMARK_CONFIGS:
        # 跳过不合理的配置
        if num_workers > num_envs:
            continue
        if num_workers > cpu_cores:
            print(
                f"Skipping config ({num_envs}, {num_workers}) as workers > available cores."
            )
            continue

        env_fns = [lambda: CartPoleEnv() for _ in range(num_envs)]
        config_str = f"({num_envs}, {num_workers})"
        print(f"{config_str:<25} | ", end="", flush=True)

        # 1. 测试 EnvExecutor (我们的)
        fps_hpe = run_benchmark(EnvExecutor, env_fns, num_workers, TOTAL_STEPS)
        print(f"{fps_hpe:18.0f} | ", end="", flush=True)

        # 2. 测试 gym.vector.AsyncVectorEnv
        fps_async = run_benchmark(AsyncVectorEnv, env_fns, num_workers, TOTAL_STEPS)
        print(f"{fps_async:20.0f} | ", end="", flush=True)

        # 3. 测试 gym.vector.SyncVectorEnv (只在第一次相关的测试中运行，因为其性能与 worker 数无关)
        fps_sync = 0
        if num_workers == BENCHMARK_CONFIGS[0][1]:  # 避免重复测试
            fps_sync = run_benchmark(SyncVectorEnv, env_fns, num_workers, TOTAL_STEPS)
            print(f"{fps_sync:18.0f}")
        else:
            print(f"{'(skipped)':>18}")

    print("-" * 88)


if __name__ == "__main__":
    main()
