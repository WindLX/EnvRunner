from copy import deepcopy

import pytest
import gymnasium as gym
import numpy as np

from envrunner.sync_vector_env import SyncSubVectorEnv


@pytest.fixture
def env_fns():
    # 使用一个简单且确定的环境进行测试
    return [lambda: gym.make("CartPole-v1") for _ in range(4)]


@pytest.fixture
def vec_env(env_fns):
    env = SyncSubVectorEnv(env_fns)
    yield env
    env.close()


def test_initialization(vec_env):
    """测试环境是否被正确初始化。"""
    assert vec_env.num_envs == 4
    assert isinstance(vec_env.observation_space, gym.spaces.Box)
    assert isinstance(vec_env.action_space, gym.spaces.Discrete)
    # 检查动作空间是否与单个环境一致
    single_env = vec_env.envs[0]
    assert vec_env.action_space == single_env.action_space


def test_reset(vec_env):
    """测试 reset 方法的完整和部分重置功能。"""
    # 1. 测试完整重置
    obs, info = vec_env.reset()
    assert obs.shape == (vec_env.num_envs,) + vec_env.single_observation_space.shape
    assert isinstance(info, dict)
    # 初始重置，info里不应该有final_observation等
    assert "final_observation" not in info
    # `_` 掩码应全部为 True
    assert info["_"].all()

    # 2. 测试部分重置
    vec_env.reset()
    # 记录重置前的观测
    pre_reset_obs, _, _, _, _ = vec_env.step(np.array([0, 0, 0, 0]))

    reset_ids = [0, 2]
    obs, info = vec_env.reset(ids=reset_ids)

    # 检查被重置的环境的观测是否改变
    # 由于环境是随机的，我们不能断言 obs[0] != pre_reset_obs[0]，
    # 但我们可以断言未被重置的环境的观测值保持不变
    assert np.array_equal(obs[1], pre_reset_obs[1])
    assert np.array_equal(obs[3], pre_reset_obs[3])

    # 检查 `_` 掩码是否正确
    expected_mask = np.array([True, False, True, False])
    assert np.array_equal(info["_"], expected_mask)


def test_step_output_shapes(vec_env):
    """测试 step 方法返回值的形状和类型。"""
    vec_env.reset()
    actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])

    obs, rewards, terminateds, truncateds, infos = vec_env.step(actions)

    assert obs.shape == (vec_env.num_envs,) + vec_env.single_observation_space.shape
    assert obs.dtype == vec_env.single_observation_space.dtype

    assert rewards.shape == (vec_env.num_envs,)
    assert rewards.dtype == np.float32

    assert terminateds.shape == (vec_env.num_envs,)
    assert terminateds.dtype == np.bool_

    assert truncateds.shape == (vec_env.num_envs,)
    assert truncateds.dtype == np.bool_

    assert isinstance(infos, dict)


def test_auto_reset_on_done(vec_env):
    """测试环境在结束后是否会自动重置。"""
    obs, _ = vec_env.reset(seed=123)  # 使用种子以获得可复现的结果
    new_obs = deepcopy(obs)
    infos = {}

    done = np.zeros(vec_env.num_envs, dtype=bool)

    # 循环直到至少有一个环境结束
    for _ in range(500):  # CartPole-v1 episode 长度上限是 500
        actions = np.array(
            [vec_env.action_space.sample() for _ in range(vec_env.num_envs)]
        )
        new_obs, _, terminateds, truncateds, infos = vec_env.step(actions)

        done = np.logical_or(terminateds, truncateds)
        if np.any(done):
            break

    assert np.any(done), "在500步内没有任何环境结束，测试无法继续"

    done_indices = np.where(done)[0]

    # 1. 检查 info 是否包含 final_observation
    assert "final_observation" in infos

    # 2. 检查结束的环境是否有 final_observation，未结束的是否为 None
    for i in range(vec_env.num_envs):
        if i in done_indices:
            assert infos["final_observation"][i] is not None
            assert (
                infos["final_observation"][i].shape
                == vec_env.single_observation_space.shape
            )
        else:
            assert infos["final_observation"][i] is None

    # 3. 记录结束环境的 "新" 观测值（来自重置后）
    obs_after_done = new_obs

    # 再执行一步
    actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])
    final_obs, _, _, _, _ = vec_env.step(actions)

    # 4. 确认被重置的环境的观测值与上一步的观测值不同
    for i in done_indices:
        # 如果自动重置成功，那么 `final_obs` 应该是 `step` 之后的结果，
        # 而 `obs_after_done` 是 `reset` 之后的结果，它们不应该相等。
        assert not np.array_equal(final_obs[i], obs_after_done[i])


# 自定义一个环境用于测试 info 聚合
class InfoTestEnv(gym.Env):
    def __init__(self, common_key_val, unique_key_val=None):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.common_key_val = common_key_val
        self.unique_key_val = unique_key_val
        self.step_count = 0

    def step(self, action):
        self.step_count += 1
        info = {"common": self.common_key_val}
        if self.unique_key_val is not None:
            info["unique"] = self.unique_key_val

        # 在第二步结束
        done = self.step_count >= 2
        return self.observation_space.sample(), 0, done, False, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.step_count = 0
        return self.observation_space.sample(), {}


def test_info_aggregation():
    """专门测试 info 聚合的逻辑。"""
    env_fns = [
        lambda: InfoTestEnv(common_key_val=10, unique_key_val="A"),
        lambda: InfoTestEnv(common_key_val=20, unique_key_val="B"),
        lambda: InfoTestEnv(common_key_val=30),  # 这个环境没有 unique_key
    ]
    env = SyncSubVectorEnv(env_fns)
    env.reset()

    # 第一步，所有环境都应该返回 info
    _, _, _, _, infos = env.step(np.array([0, 0, 0]))

    # 测试 common key，所有环境都有
    assert "common" in infos
    assert np.array_equal(infos["common"], np.array([10, 20, 30]))

    # 测试 unique key，部分环境有
    assert "unique" in infos
    assert np.array_equal(infos["unique"], np.array(["A", "B", None], dtype=object))

    # 第二步，所有环境都将结束
    _, _, terminateds, _, infos = env.step(np.array([0, 0, 0]))

    assert terminateds.all()
    assert "final_observation" in infos
    # 检查 final_observation 是否也正确地聚合了 (所有环境都应该有)
    assert (
        infos["final_observation"].shape
        == (env.num_envs,) + env.single_observation_space.shape  # type: ignore
    )
    assert None not in infos["final_observation"]

    env.close()
