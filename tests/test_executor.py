import pytest

from typing import Any, Generator
import gymnasium as gym
import numpy as np

from envrunner.env_executor import EnvExecutor
from envrunner.types import AutoResetMode

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


def test_no_autoreset_behavior(executor_config: tuple[int, int]):
    total_envs, num_workers = executor_config
    max_episode_steps = 501

    env_fns = [lambda: gym.make("CartPole-v1") for _ in range(total_envs)]

    # 注意：根据上一轮的实现，参数名是 autoreset (bool)，如果你封装了 Enum 请自行转换
    env = EnvExecutor(
        env_fns, num_workers=num_workers, autoreset_mode=AutoResetMode.DISABLE
    )

    env.reset(seed=123)

    # 这个 mask 用于在测试端记录：哪些环境已经结束了但还没被我们手动 reset
    # 初始全为 False (都活着)
    pending_reset_mask = np.zeros(total_envs, dtype=bool)

    # 记录是否所有环境都至少结束过一次（为了验证测试的完整性）
    all_time_done_mask = np.zeros(total_envs, dtype=bool)

    for step_i in range(max_episode_steps):
        actions = np.array([env.action_space.sample() for _ in range(total_envs)])

        # obs, rewards, term, trunc, info
        _, rewards, terminateds, truncateds, infos = env.step(actions)

        # 当前这一步刚刚结束的环境
        just_finished = np.logical_or(terminateds, truncateds)

        # 更新全时段记录
        all_time_done_mask = np.logical_or(all_time_done_mask, just_finished)

        # 验证逻辑 1:
        # 如果环境之前已经是 pending_reset 状态，那么这一步它应该返回占位符数据
        # 即: term/trunc 应该为 False (因为没运行), reward 应该为 0
        already_dead_indices = np.where(pending_reset_mask)[0]
        if len(already_dead_indices) > 0:
            assert not np.any(
                terminateds[already_dead_indices]
            ), "已停止的环境不应再次返回 terminated=True"
            assert not np.any(
                truncateds[already_dead_indices]
            ), "已停止的环境不应再次返回 truncated=True"
            assert np.all(
                rewards[already_dead_indices] == 0
            ), "已停止的环境 reward 应为 0"

        # 更新当前处于等待 reset 状态的 mask
        # 逻辑：旧的等待者 + 新的结束者 = 现在的等待者
        pending_reset_mask = np.logical_or(pending_reset_mask, just_finished)

        # 验证逻辑 2:
        # 环境返回的 _reset_mask 必须完全等于我们自己推算的 pending_reset_mask
        assert "_reset_mask" in infos
        env_reset_mask = infos["_reset_mask"]

        # 使用 np.array_equal 进行整体比较，或者逐个元素比较
        np.testing.assert_array_equal(
            env_reset_mask,
            pending_reset_mask,
            err_msg=f"Step {step_i}: 环境返回的 reset_mask 与预期不符",
        )

        # (可选) 模拟 Evaluation 过程：如果所有环境都结束了，我们可以选择全部 reset 或者部分 reset
        # 为了测试持续运行，我们可以在这里不 reset，一直等到所有都跑完，
        # 或者随机 reset 一些（如下所示）：
        if np.any(pending_reset_mask):
            ids_to_reset = np.where(pending_reset_mask)[0]
            # 演示：只重置前一半已结束的环境，另一半继续保持 done 状态
            ids_to_reset = ids_to_reset[: len(ids_to_reset) // 2]
            if len(ids_to_reset) > 0:
                env.reset(ids=ids_to_reset)
                pending_reset_mask[ids_to_reset] = False  # 测试端同步更新状态

    assert np.all(
        all_time_done_mask
    ), f"并非所有环境都在 {max_episode_steps} 步内结束。"

    env.close()
