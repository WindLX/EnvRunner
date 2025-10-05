import pytest
from pathlib import Path
import numpy as np
import gymnasium as gym
from typing import Any, Dict, Tuple

from envrunner import VecEnv, VecObsNormalizer


# --- 模拟环境 ---
class MockVecEnv(VecEnv):
    """一个模拟的矢量化环境，用于测试包装器。"""

    def __init__(self, num_envs: int, obs_shape: Tuple[int, ...], action_dim: int):
        self.num_envs = num_envs
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=obs_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        self._is_closed = False
        self._step_count = 0

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._step_count = 0
        # 返回一个确定的、非零的观测值，方便测试
        assert self.observation_space.shape is not None
        return (
            np.ones((self.num_envs,) + self.observation_space.shape, dtype=np.float32),
            {},
        )

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        self._step_count += 1
        # 返回一个与reset不同的观测值
        assert self.observation_space.shape is not None
        obs = np.full(
            (self.num_envs,) + self.observation_space.shape,
            self._step_count + 1,
            dtype=np.float32,
        )
        rewards = np.ones(self.num_envs, dtype=np.float32)
        terminateds = np.zeros(self.num_envs, dtype=np.bool_)
        truncateds = np.zeros(self.num_envs, dtype=np.bool_)
        return obs, rewards, terminateds, truncateds, {}

    def close(self) -> None:
        self._is_closed = True


# --- Pytest Fixtures ---
@pytest.fixture
def mock_vec_env_simple() -> MockVecEnv:
    """提供一个简单的矢量观测环境。"""
    return MockVecEnv(num_envs=2, obs_shape=(4,), action_dim=1)


@pytest.fixture
def mock_vec_env_sequence() -> MockVecEnv:
    """提供一个带有序列维度的观测环境。"""
    return MockVecEnv(num_envs=2, obs_shape=(10, 8), action_dim=1)


# --- 测试用例 ---
class TestVecObsNormalizer:

    def test_initialization_and_proxy(self, mock_vec_env_simple: MockVecEnv):
        """测试初始化和属性代理。"""
        wrapper = VecObsNormalizer(mock_vec_env_simple)
        assert wrapper.num_envs == mock_vec_env_simple.num_envs
        assert wrapper.observation_space == mock_vec_env_simple.observation_space

        # 测试方法代理
        assert not mock_vec_env_simple._is_closed
        wrapper.close()
        assert mock_vec_env_simple._is_closed

    def test_reset_and_step_normalization(self, mock_vec_env_simple: MockVecEnv):
        """测试 reset 和 step 是否正确归一化了观测值。"""
        wrapper = VecObsNormalizer(mock_vec_env_simple)

        # 测试 reset
        raw_obs, _ = mock_vec_env_simple.reset()
        normalized_obs, _ = wrapper.reset()

        expected_shape = (
            mock_vec_env_simple.num_envs,  # type: ignore
        ) + mock_vec_env_simple.observation_space.shape
        assert normalized_obs.shape == expected_shape

        # 归一化后的值不应等于原始值（原始值为1）
        assert not np.allclose(normalized_obs, raw_obs)

        # 测试 step
        actions = np.zeros((mock_vec_env_simple.num_envs, 1))
        raw_next_obs, *_ = mock_vec_env_simple.step(actions)
        normalized_next_obs, *_ = wrapper.step(actions)
        assert normalized_next_obs.shape == raw_next_obs.shape
        assert not np.allclose(normalized_next_obs, raw_next_obs)

    def test_training_flag(self, mock_vec_env_simple: MockVecEnv):
        """测试 training 标志是否控制统计数据的更新。"""
        # 训练模式
        wrapper_train = VecObsNormalizer(mock_vec_env_simple, training=True)
        initial_mean_train = wrapper_train.obs_rms.mean.copy()
        wrapper_train.reset()
        # 均值应该已经更新，不再是初始的零
        assert not np.allclose(wrapper_train.obs_rms.mean, initial_mean_train)

        # 推理模式
        wrapper_eval = VecObsNormalizer(mock_vec_env_simple, training=False)
        initial_mean_eval = wrapper_eval.obs_rms.mean.copy()
        wrapper_eval.reset()
        # 均值不应更新，保持为初始的零
        assert np.allclose(wrapper_eval.obs_rms.mean, initial_mean_eval)

    def test_save_and_load_stats(self, mock_vec_env_simple: MockVecEnv, tmp_path: Path):
        """测试统计数据的保存和加载。"""
        stats_file = tmp_path / "obs_stats.npz"

        wrapper1 = VecObsNormalizer(mock_vec_env_simple)
        wrapper1.reset()
        wrapper1.step(np.zeros((2, 1)))
        wrapper1.save_stats(str(stats_file))

        # 创建一个新的包装器
        wrapper2 = VecObsNormalizer(mock_vec_env_simple)
        # 确认其均值是初始值（零）
        assert np.all(wrapper2.obs_rms.mean == 0)

        # 加载统计数据
        wrapper2.load_stats(str(stats_file))

        # 确认统计数据已恢复
        assert np.allclose(wrapper1.obs_rms.mean, wrapper2.obs_rms.mean)
        assert np.allclose(wrapper1.obs_rms.var, wrapper2.obs_rms.var)

    def test_sequence_normalization_broadcast(self, mock_vec_env_sequence: MockVecEnv):
        """测试序列数据归一化是否正确广播。"""
        wrapper = VecObsNormalizer(mock_vec_env_sequence)

        # 第一次 reset，观测值为1，归一化后为0
        wrapper.reset()
        # 第二次 step，观测值为2，此时均值和标准差都会更新为非零值
        normalized_obs, _, _, _, _ = wrapper.step(
            np.zeros((mock_vec_env_sequence.num_envs, 1))
        )

        # 修复: 验证批量观测的形状
        expected_shape = (
            mock_vec_env_sequence.num_envs,  # type: ignore
        ) + mock_vec_env_sequence.observation_space.shape
        assert normalized_obs.shape == expected_shape

        # 检查特征维度为8的 RunningMeanStd 是否被正确广播
        # 对于同一个特征（例如第3个特征），其在所有时间步上的值应该相同
        for feature_idx in range(normalized_obs.shape[-1]):
            # 取出第一个环境，第 feature_idx 个特征的所有时间步的值
            feature_over_time = normalized_obs[0, :, feature_idx]
            # 确认这个特征在所有时间步的值都相等
            assert np.allclose(feature_over_time, feature_over_time[0])
            # 确认这个值不是0，证明我们测试的是非退化的情况
            assert not np.isclose(feature_over_time[0], 0.0)

    def test_selective_normalization_slice(self, mock_vec_env_sequence: MockVecEnv):
        """测试使用 slice 进行选择性归一化。"""
        # 只归一化前4个特征 (索引 0, 1, 2, 3)
        keys_to_norm = slice(0, 4)
        wrapper = VecObsNormalizer(mock_vec_env_sequence, norm_obs_keys=keys_to_norm)

        normalized_obs, _ = wrapper.reset()
        raw_obs, _ = mock_vec_env_sequence.reset()  # 原始观测值为1

        # 被归一化的部分 (前4个特征)
        normalized_part = normalized_obs[..., keys_to_norm]
        raw_part_norm = raw_obs[..., keys_to_norm]
        assert not np.allclose(normalized_part, raw_part_norm)

        # 未被归一化的部分 (后4个特征)
        unnormalized_part = normalized_obs[..., 4:]
        raw_part_unnorm = raw_obs[..., 4:]
        assert np.allclose(unnormalized_part, raw_part_unnorm)

    def test_selective_normalization_indices(self, mock_vec_env_sequence: MockVecEnv):
        """测试使用索引列表进行选择性归一化。"""
        keys_to_norm = [0, 3, 5]
        keys_not_to_norm = [1, 2, 4, 6, 7]
        wrapper = VecObsNormalizer(mock_vec_env_sequence, norm_obs_keys=keys_to_norm)

        normalized_obs, _ = wrapper.reset()
        raw_obs, _ = mock_vec_env_sequence.reset()  # 原始观测值为1

        # 被归一化的部分
        normalized_part = normalized_obs[..., keys_to_norm]
        raw_part_norm = raw_obs[..., keys_to_norm]
        assert not np.allclose(normalized_part, raw_part_norm)

        # 未被归一化的部分
        unnormalized_part = normalized_obs[..., keys_not_to_norm]
        raw_part_unnorm = raw_obs[..., keys_not_to_norm]
        assert np.allclose(unnormalized_part, raw_part_unnorm)

    def test_clipping(self, mock_vec_env_simple: MockVecEnv):
        """测试观测值裁剪功能。"""
        wrapper = VecObsNormalizer(mock_vec_env_simple, clip_obs=0.5)

        # 手动设置一个会导致归一化后值很大的均值和方差
        # 原始观测值为1, mean=0, std=0.1, 归一化后为 (1-0)/0.1 = 10
        assert mock_vec_env_simple.observation_space.shape is not None
        wrapper.obs_rms.mean = np.zeros(mock_vec_env_simple.observation_space.shape)
        wrapper.obs_rms.var = np.full(mock_vec_env_simple.observation_space.shape, 0.01)

        normalized_obs, _ = wrapper.reset()

        # 确认所有值都被裁剪到了 [-0.5, 0.5] 范围内
        assert np.max(normalized_obs) <= 0.5
        assert np.min(normalized_obs) >= -0.5
