from typing import Any, Sequence
from pathlib import Path

import numpy as np
import gymnasium as gym

from ..base import VecEnv


class RunningMeanStd:
    """计算运行时的均值和方差，支持选择性更新。"""

    def __init__(self, shape: tuple[int, ...], epsilon: float = 1e-8):
        self.mean: np.ndarray = np.zeros(shape, dtype=np.float64)
        self.var: np.ndarray = np.ones(shape, dtype=np.float64)
        self.count: float = epsilon
        self.epsilon: float = epsilon

    def update(self, x: np.ndarray) -> None:
        """根据一批新的数据更新均值和方差。"""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(
        self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int
    ) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + self.epsilon)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def unnormalize(self, x_norm: np.ndarray) -> np.ndarray:
        return x_norm * self.std + self.mean


class VecObsNormalizer:
    """
    为矢量化环境执行器提供集中的、可选择维度的、运行时的观测归一化。

    它通过组合的方式工作，并能正确处理时间序列结构的观测数据。
    """

    def __init__(
        self,
        vec_env: VecEnv,
        training: bool = True,
        clip_obs: float = 10.0,
        norm_obs_keys: slice | Sequence[int] | None = None,
    ):
        """
        Args:
            vec_env: 一个实现了 VecEnv 协议的矢量化环境执行器实例。
            training (bool): 是否更新运行时的均值和标准差。部署时应设为 False。
            clip_obs (float): 归一化后观测值的裁剪范围（正负）。
            norm_obs_keys (Optional[Union[slice, Sequence[int]]]):
                一个索引序列或切片，用于指定在最后一个维度上要归一化的特征。
                如果为 None，则归一化所有特征。
                例如:
                - slice(0, 58) 表示归一化前58个特征。
                - [0, 1, 2, 5, 8] 表示归一化指定的几个特征。
        """
        self._vec_env: VecEnv = vec_env
        self.training: bool = training
        self.clip_obs: float = clip_obs

        # 从内部环境推断空间
        self.observation_space: gym.spaces.Space = self._vec_env.observation_space
        self.action_space: gym.spaces.Space = self._vec_env.action_space
        self.num_envs: int = self._vec_env.num_envs

        if not isinstance(self.observation_space, gym.spaces.Box):
            raise TypeError("VecObsNormalizer只支持Box类型的观测空间。")

        # 处理要归一化的维度
        self.norm_obs_keys: slice | np.ndarray | None
        if norm_obs_keys is None:
            # 默认归一化所有维度
            self.norm_obs_keys = slice(None)
            feature_shape = self.observation_space.shape[-1:]
        elif isinstance(norm_obs_keys, slice):
            self.norm_obs_keys = norm_obs_keys
            # 计算切片后的特征数量
            start = norm_obs_keys.start or 0
            stop = norm_obs_keys.stop or self.observation_space.shape[-1]
            step = norm_obs_keys.step or 1
            feature_shape = (len(range(start, stop, step)),)
        else:  # Sequence[int]
            self.norm_obs_keys = np.array(norm_obs_keys, dtype=np.intp)
            feature_shape = self.norm_obs_keys.shape

        # 初始化运行时的统计对象，其形状只包含特征维度
        self.obs_rms: RunningMeanStd = RunningMeanStd(shape=feature_shape)

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """内部方法，用于更新统计数据和归一化选择的观测维度。"""
        # 提取需要归一化的部分
        # obs 的形状可能是 (num_envs, obs_dim) 或 (num_envs, seq_len, obs_dim)
        obs_to_norm = obs[..., self.norm_obs_keys]

        if self.training:
            # 更新统计数据时，需要将所有环境和时间步的数据展平
            # 例如，(num_envs, seq_len, feature_dim) -> (num_envs * seq_len, feature_dim)
            if obs_to_norm.ndim > 2:
                reshaped_obs = obs_to_norm.reshape(-1, obs_to_norm.shape[-1])
                self.obs_rms.update(reshaped_obs)
            else:
                self.obs_rms.update(obs_to_norm)

        # 进行归一化
        # self.obs_rms.normalize 会自动利用 numpy 的广播机制
        # (num_envs, seq_len, feature_dim) - (feature_dim,) -> (num_envs, seq_len, feature_dim)
        normalized_part = self.obs_rms.normalize(obs_to_norm)
        clipped_part = np.clip(normalized_part, -self.clip_obs, self.clip_obs)

        # 将归一化后的部分放回原位
        # 创建一个 obs 的副本以避免修改原始数据（如果它被其他地方使用）
        obs_copy = obs.copy()
        obs_copy[..., self.norm_obs_keys] = clipped_part

        return obs_copy

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """调用内部环境的 reset，然后归一化返回的观测值。"""
        raw_obs, info = self._vec_env.reset(**kwargs)
        normalized_obs = self._normalize_obs(raw_obs)
        return normalized_obs, info

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """将动作传递给内部环境，然后归一化返回的下一步观测值。"""
        raw_next_obs, rewards, terminateds, truncateds, infos = self._vec_env.step(
            actions
        )
        normalized_next_obs = self._normalize_obs(raw_next_obs)
        return normalized_next_obs, rewards, terminateds, truncateds, infos

    def save_stats(self, path: Path | str) -> None:
        """保存运行时的统计数据到一个文件。"""
        np.savez(
            path,
            obs_mean=self.obs_rms.mean,
            obs_var=self.obs_rms.var,
            obs_count=self.obs_rms.count,
        )
        print(f"观测归一化统计数据已保存到: {path}")

    def load_stats(self, path: Path | str) -> None:
        """从文件加载运行时的统计数据。"""
        try:
            with np.load(path) as data:
                # 验证加载的形状是否与期望的特征形状匹配
                expected_shape = self.obs_rms.mean.shape
                if data["obs_mean"].shape != expected_shape:
                    raise ValueError(
                        f"加载的均值形状 {data['obs_mean'].shape} 与期望的形状 {expected_shape} 不匹配。"
                    )

                self.obs_rms.mean = data["obs_mean"]
                self.obs_rms.var = data["obs_var"]
                self.obs_rms.count = data["obs_count"]
            print(f"成功从 {path} 加载观测归一化统计数据。")
        except FileNotFoundError:
            print(f"警告: 找不到统计文件 {path}。将使用默认的初始统计数据。")
        except Exception as e:
            print(f"加载统计数据时出错: {e}")

    def close(self) -> None:
        """关闭内部环境。"""
        self._vec_env.close()

    def __getattr__(self, name: str) -> Any:
        """代理对内部环境的其他属性的访问。"""
        if name.startswith("_"):
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
        return getattr(self._vec_env, name)
