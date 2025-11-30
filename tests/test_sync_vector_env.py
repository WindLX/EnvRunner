import pytest

from typing import Callable, Generator

import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import numpy as np

from envrunner.sync_vector_env import SyncSubVectorEnv
from envrunner.types import AutoResetMode


# --- Fixtures ---
@pytest.fixture
def env_fns_simple() -> list[Callable[[], gym.Env]]:
    """返回4个简单的 CartPole 环境构造函数。"""
    return [lambda: gym.make("CartPole-v1") for _ in range(4)]


@pytest.fixture
def env_fns_with_stats() -> list[Callable[[], gym.Env]]:
    """返回4个被 RecordEpisodeStatistics 包装的环境构造函数。"""
    # 这个 wrapper 会在 episode 结束时在 info 中添加 "episode" 键
    return [lambda: RecordEpisodeStatistics(gym.make("CartPole-v1")) for _ in range(4)]


@pytest.fixture
def vec_env(
    env_fns_simple: list[Callable[[], gym.Env]],
) -> Generator[SyncSubVectorEnv, None, None]:
    """创建一个基础的 SyncSubVectorEnv 实例。"""
    env = SyncSubVectorEnv(env_fns_simple)
    yield env
    env.close()


# --- 测试用例 ---
def test_initialization(vec_env: SyncSubVectorEnv):
    """测试初始化是否正确。"""
    assert vec_env.num_envs == 4
    # 检查空间是否被正确向量化 (这里我们不测试 _create_vectorized_space 的所有细节)
    assert vec_env.observation_space.shape[0] == 4  # type: ignore[attr-defined]
    assert vec_env.action_space.shape is not None  # Box or MultiDiscrete
    assert len(vec_env.envs) == 4


def test_reset_full(vec_env: SyncSubVectorEnv):
    """测试完整的 reset。"""
    obs, info = vec_env.reset(seed=42)

    assert obs.shape == (4, 4)  # (num_envs, obs_dim) for CartPole
    assert isinstance(info, dict)
    assert "_" in info
    assert info["_"].all()  # `_` 掩码应全部为 True
    assert len(info) == 1  # 初始 reset 的 info 应该只包含 `_`


def test_reset_partial_by_list(vec_env: SyncSubVectorEnv):
    """测试通过索引列表进行部分 reset。"""
    vec_env.reset(seed=42)
    # 走一步，让环境状态发生变化
    pre_reset_obs, _, _, _, _ = vec_env.step(np.array([0] * 4))

    reset_ids = [0, 3]
    obs, info = vec_env.reset(ids=reset_ids, seed=123)

    # 检查返回的 obs 形状
    assert obs.shape == pre_reset_obs.shape

    # 检查未被重置的环境的观测值是否保持不变
    assert np.array_equal(obs[1], pre_reset_obs[1])
    assert np.array_equal(obs[2], pre_reset_obs[2])

    # 检查被重置的环境的观测值是否已改变
    assert not np.array_equal(obs[0], pre_reset_obs[0])
    assert not np.array_equal(obs[3], pre_reset_obs[3])

    # 检查 `_` 掩码是否正确
    expected_mask = np.array([True, False, False, True])
    assert np.array_equal(info["_"], expected_mask)


def test_reset_partial_by_mask(vec_env: SyncSubVectorEnv):
    """测试通过布尔掩码进行部分 reset。"""
    vec_env.reset(seed=42)
    pre_reset_obs, _, _, _, _ = vec_env.step(np.array([0] * 4))

    reset_mask = np.array([False, True, True, False])
    obs, info = vec_env.reset(ids=reset_mask, seed=123)

    assert np.array_equal(obs[0], pre_reset_obs[0])
    assert not np.array_equal(obs[1], pre_reset_obs[1])
    assert not np.array_equal(obs[2], pre_reset_obs[2])
    assert np.array_equal(obs[3], pre_reset_obs[3])

    assert np.array_equal(info["_"], reset_mask)


def test_step_outputs(vec_env: SyncSubVectorEnv):
    """测试 step 方法返回值的形状和类型。"""
    vec_env.reset()
    actions = np.array([vec_env.single_action_space.sample() for _ in range(4)])

    obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

    assert obs.shape == (4, 4)
    assert rewards.shape == (4,)
    assert terminateds.shape == (4,)
    assert truncateds.shape == (4,)

    assert rewards.dtype == np.float32
    assert terminateds.dtype == np.bool_
    assert truncateds.dtype == np.bool_

    assert isinstance(infos, dict)


def test_auto_reset_and_final_observation(env_fns_simple: list[Callable[[], gym.Env]]):
    """测试当环境结束时，是否会自动重置并提供 final_observation。"""
    vec_env = SyncSubVectorEnv(env_fns_simple)
    vec_env.reset(seed=123)
    dones = np.array([False] * 4)
    infos = {}

    # 循环足够多的步数以确保至少有一个环境结束
    for _ in range(501):  # CartPole-v1 max steps is 500
        actions = np.array([0] * 4)  # 保持一个动作，加速结束
        obs, _, terminateds, truncateds, infos = vec_env.step(actions)
        dones = np.logical_or(terminateds, truncateds)
        if np.any(dones):
            break

    assert np.any(dones), "在501步内没有任何环境结束"

    done_indices = np.where(dones)[0]

    # 1. 检查 `final_observation` 键是否存在
    assert "final_observation" in infos

    # 2. 检查聚合后的 `final_observation` 数组
    final_obs_array = infos["final_observation"]
    assert final_obs_array.dtype == object
    assert final_obs_array.shape == (4,)

    for i in range(4):
        if i in done_indices:
            assert final_obs_array[i] is not None
            assert final_obs_array[i].shape == vec_env.single_observation_space.shape
        else:
            assert final_obs_array[i] is None

    vec_env.close()


def test_final_info_with_wrapper(env_fns_with_stats: list[Callable[[], gym.Env]]):
    """
    这是一个关键测试：验证 RecordEpisodeStatistics 的信息是否被正确捕获到 final_info 中。
    """
    vec_env = SyncSubVectorEnv(env_fns_with_stats)
    vec_env.reset(seed=123)
    dones = np.array([False] * 4)
    infos = {}

    for _ in range(501):
        actions = np.array([vec_env.single_action_space.sample() for _ in range(4)])
        obs, _, terminateds, truncateds, infos = vec_env.step(actions)
        dones = np.logical_or(terminateds, truncateds)
        if np.any(dones):
            break

    assert np.any(dones), "在501步内没有任何环境结束"
    done_indices = np.where(dones)[0]

    # 1. 检查 `final_info` 键
    assert "final_info" in infos
    final_info_array = infos["final_info"]
    assert final_info_array.dtype == object

    # 2. 检查内容
    for i in range(4):
        if i in done_indices:
            final_info = final_info_array[i]
            assert isinstance(final_info, dict)
            # RecordEpisodeStatistics 的关键输出
            assert "episode" in final_info
            assert "r" in final_info["episode"]  # reward
            assert "l" in final_info["episode"]  # length
            assert "t" in final_info["episode"]  # time
        else:
            assert final_info_array[i] is None

    vec_env.close()


@pytest.fixture
def vec_env_noautoreset(
    env_fns_simple: list[Callable[[], gym.Env]],
) -> Generator[SyncSubVectorEnv, None, None]:
    """创建一个基础的 SyncSubVectorEnv 实例。"""
    env = SyncSubVectorEnv(env_fns_simple, autoreset_mode=AutoResetMode.DISABLE)
    yield env
    env.close()


def test_no_autoreset_behavior(vec_env_noautoreset: SyncSubVectorEnv):
    """测试在禁用自动重置时，环境结束的处理行为。"""
    vec_env = vec_env_noautoreset
    vec_env.reset(seed=123)
    dones = np.array([False] * 4)
    infos = {}

    # 循环足够多的步数以确保至少有一个环境结束
    for _ in range(501):  # CartPole-v1 max steps is 500
        actions = np.array([0] * 4)  # 保持一个动作，加速结束
        _, _, terminateds, truncateds, infos = vec_env.step(actions)
        dones = np.logical_or(terminateds, truncateds)
        if np.any(dones):
            break

    assert np.any(dones), "在501步内没有任何环境结束"

    done_indices = np.where(dones)[0]

    assert "_reset_mask" in infos

    reset_mask_array = infos["_reset_mask"]
    assert reset_mask_array.dtype == bool
    assert reset_mask_array.shape == (4,)

    for i in range(4):
        if i in done_indices:
            assert reset_mask_array[i] == True
        else:
            assert reset_mask_array[i] == False

    vec_env.close()
