import pytest

from typing import Any, Generator
import gymnasium as gym
import numpy as np

from envrunner.env_executor import EnvExecutor

# 定义测试参数：(环境总数, worker数量)
# 确保环境总数可以被 worker 数量整除
TEST_CONFIGS = [
    (8, 2),  # 8个环境，2个worker，每个worker 4个环境
    (4, 4),  # 4个环境，4个worker，每个worker 1个环境
    (12, 3),  # 12个环境，3个worker，每个worker 4个环境
]


@pytest.fixture(scope="module", params=TEST_CONFIGS)
def executor_config(request: pytest.FixtureRequest) -> tuple[int, int]:
    """提供环境总数和worker数量的配置。"""
    return request.param


@pytest.fixture
def executor(executor_config: tuple[int, int]) -> Generator[EnvExecutor, None, None]:
    """创建一个 HighPerformanceEnvExecutor 实例并确保在测试后关闭。"""
    total_envs, num_workers = executor_config
    env_fns = [lambda: gym.make("CartPole-v1") for _ in range(total_envs)]

    # 确保在测试开始前没有残留的共享内存块
    # 注意：这在实际测试中可能需要更复杂的清理逻辑

    env = EnvExecutor(env_fns, num_workers=num_workers)
    yield env
    env.close()


def test_initialization(executor: EnvExecutor, executor_config: tuple[int, int]):
    """测试执行器是否正确初始化。"""
    total_envs, num_workers = executor_config
    assert executor.num_envs == total_envs
    assert executor.num_workers == num_workers
    assert not executor.closed
    assert isinstance(executor.observation_space, gym.spaces.Box)
    assert isinstance(executor.action_space, gym.spaces.Discrete)


def test_reset(executor: EnvExecutor):
    """测试 reset 方法。"""
    obs, info = executor.reset()
    assert obs.shape == (executor.num_envs,) + executor.single_observation_space.shape  # type: ignore
    assert isinstance(info, dict)
    # 初始重置不应包含 final_* 信息
    assert "final_observation" not in info


def test_step(executor: EnvExecutor):
    """测试 step 方法的返回形状和类型。"""
    executor.reset()
    actions = np.array(
        [executor.action_space.sample() for _ in range(executor.num_envs)]
    )

    obs, rewards, terminateds, truncateds, infos = executor.step(actions)

    assert obs.shape == (executor.num_envs,) + executor.single_observation_space.shape  # type: ignore
    assert rewards.shape == (executor.num_envs,)
    assert terminateds.shape == (executor.num_envs,)
    assert truncateds.shape == (executor.num_envs,)

    assert obs.dtype == executor.single_observation_space.dtype
    assert rewards.dtype == np.float32
    assert terminateds.dtype == np.bool_
    assert truncateds.dtype == np.bool_
    assert isinstance(infos, dict)


def test_auto_reset_and_final_info(executor_config: tuple[int, int]):
    """
    一个更复杂的测试，验证跨 worker 的自动重置和 final_observation 聚合。
    """
    total_envs, num_workers = executor_config
    # CartPole-v1 的 episode 长度上限是 500
    max_episode_steps = 501

    env_fns = [lambda: gym.make("CartPole-v1") for _ in range(total_envs)]
    env = EnvExecutor(env_fns, num_workers=num_workers)

    env.reset(seed=123)

    done_mask = np.zeros(total_envs, dtype=bool)

    for _ in range(max_episode_steps):
        actions = np.array([env.action_space.sample() for _ in range(total_envs)])
        _, _, terminateds, truncateds, infos = env.step(actions)

        current_dones = np.logical_or(terminateds, truncateds)
        if np.any(current_dones):
            done_mask = np.logical_or(done_mask, current_dones)

            # 验证 final_observation
            assert "final_observation" in infos
            final_obs_array = infos["final_observation"]

            for i in range(total_envs):
                if current_dones[i]:
                    # 结束的环境必须有 final_observation
                    assert final_obs_array[i] is not None
                    assert (
                        final_obs_array[i].shape == env.single_observation_space.shape
                    )
                else:
                    # 未结束的环境的 final_observation 应该是 None
                    assert final_obs_array[i] is None

    # 确保在这么多步之后，所有环境都至少重置过一次
    assert np.all(done_mask), f"并非所有环境都在 {max_episode_steps} 步内结束。"

    env.close()


def test_close_idempotency():
    """测试 close 方法可以被安全地多次调用。"""
    env_fns = [lambda: gym.make("CartPole-v1") for _ in range(2)]
    env = EnvExecutor(env_fns, num_workers=1)

    assert not env.closed
    env.close()
    assert env.closed

    # 再次调用 close 不应该抛出异常
    try:
        env.close()
    except Exception as e:
        pytest.fail(f"第二次调用 close() 时引发异常: {e}")

    # 在已关闭的环境上操作应引发异常
    with pytest.raises(gym.error.ClosedEnvironmentError):
        env.reset()

    with pytest.raises(gym.error.ClosedEnvironmentError):
        actions = np.array([env.action_space.sample() for _ in range(2)])
        env.step(actions)


# --- 测试错误处理的辅助环境 ---
class CrashingEnv(gym.Env):
    """一个在特定步骤会崩溃的环境。"""

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        self.action_space = gym.spaces.Discrete(2)
        self.step_count = 0

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.step_count += 1
        if self.step_count >= 3:
            raise ValueError("模拟的环境崩溃!")
        return np.array([0.0]), 0, False, False, {}

    def reset(self, *, seed=None, options=None):
        self.step_count = 0
        return np.array([0.0]), {}


def test_worker_crash_handling():
    """测试当一个 worker 内部的环境崩溃时，主进程是否能捕获错误。"""
    # 让第2个环境崩溃 (假设有4个环境，2个worker，崩溃发生在worker 0)
    env_fns = [
        lambda: gym.make("CartPole-v1"),
        lambda: CrashingEnv(),
        lambda: gym.make("CartPole-v1"),
        lambda: gym.make("CartPole-v1"),
    ]

    env = EnvExecutor(env_fns, num_workers=2)
    env.reset()
    env.step(np.array([0, 0, 0, 0]))  # Step 1
    env.step(np.array([0, 0, 0, 0]))  # Step 2

    # 在 Step 3，CrashingEnv 将会崩溃
    with pytest.raises(RuntimeError, match="模拟的环境崩溃!"):
        env.step(np.array([0, 0, 0, 0]))

    # 发生错误后，执行器应该被自动关闭
    assert env.closed


def test_multiple_instances():
    """测试同时创建和运行两个执行器实例。"""
    env_fns1 = [lambda: gym.make("CartPole-v1") for _ in range(4)]
    env_fns2 = [lambda: gym.make("MountainCarContinuous-v0") for _ in range(2)]

    executor1 = None
    executor2 = None

    try:
        executor1 = EnvExecutor(env_fns1, num_workers=2)
        executor2 = EnvExecutor(env_fns2, num_workers=2)

        assert executor1.num_envs == 4
        assert executor2.num_envs == 2

        # 验证共享内存名称不冲突
        assert executor1.sm_meta["obs"][0] != executor2.sm_meta["obs"][0]

        # 对两个执行器进行操作
        obs1, _ = executor1.reset()
        obs2, _ = executor2.reset()

        assert obs1.shape[0] == 4
        assert obs2.shape[0] == 2

        actions1 = np.array([executor1.action_space.sample() for _ in range(4)])
        executor1.step(actions1)

        actions2 = np.array([executor2.action_space.sample() for _ in range(2)])
        executor2.step(actions2)

    finally:
        if executor1:
            executor1.close()
        if executor2:
            executor2.close()
