from typing import Callable, Any, Sequence

import gymnasium as gym
import numpy as np


class SyncSubVectorEnv:
    """
    一个在单个进程内同步、串行执行多个环境的向量化环境。
    它的行为旨在模仿 gymnasium.vector.SyncVectorEnv，但为我们后续的
    高性能执行器作为基础组件。

    Args:
        env_fns (Sequence[Callable[[], gym.Env]]): 一个环境构造函数列表。
    """

    def __init__(self, env_fns: Sequence[Callable[[], gym.Env]]):
        if not callable(env_fns[0]):
            raise TypeError(
                "env_fns 必须是一个可调用对象的列表 (e.g., [lambda: gym.make('CartPole-v1')])"
            )

        self.env_fns = env_fns
        self.num_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]

        # 从第一个环境中推断空间
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

        # 定义向量化的观测空间
        self.observation_space = self._create_vectorized_space(
            self.single_observation_space, self.num_envs
        )

        # 向量化环境的动作空间与单个环境的动作空间结构相同
        self.action_space = self.single_action_space

        assert self.single_action_space.shape is not None, "仅支持固定形状的动作空间。"
        assert (
            self.single_observation_space.shape is not None
        ), "仅支持固定形状的观测空间。"

        self._obs_buf = np.zeros(
            (self.num_envs,) + self.single_observation_space.shape,
            dtype=self.single_observation_space.dtype,
        )
        self._rewards_buf = np.zeros((self.num_envs,), dtype=np.float32)
        self._terminateds_buf = np.zeros((self.num_envs,), dtype=np.bool_)
        self._truncateds_buf = np.zeros((self.num_envs,), dtype=np.bool_)

        # _autoreset_flags 用于标记在下一步中需要自动重置的环境
        self._autoreset_flags = np.zeros((self.num_envs,), dtype=np.bool_)

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
        ids: np.ndarray | list[int] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        重置环境。

        Args:
            seed: 用于环境重置的种子。
            options: 传递给 env.reset 的额外选项。
            ids: 指定要重置的环境。None=全部, List[int]=索引列表, np.ndarray(bool)=掩码。

        Returns:
            (observations, infos)
        """
        if ids is None:
            ids = np.arange(self.num_envs)
        elif isinstance(ids, list):
            ids = np.array(ids, dtype=int)
        elif isinstance(ids, np.ndarray) and ids.dtype == np.bool:
            ids = np.where(ids)[0]

        if isinstance(seed, int):
            seeds = [seed + int(i) for i in ids]
        elif isinstance(seed, list):
            if len(seed) != len(ids):
                raise ValueError(
                    f"提供的种子数量 ({len(seed)}) 与要重置的环境数量 ({len(ids)}) 不匹配。"
                )
            seeds = seed
        else:
            seeds = [None] * len(ids)

        infos = {}
        for i, env_idx in enumerate(ids):
            current_seed = seeds[i]
            obs, info = self.envs[env_idx].reset(seed=current_seed, options=options)
            self._obs_buf[env_idx] = obs
            # 我们将每个环境的 info 存储在其自己的键下，以便稍后聚合
            infos[f"_{env_idx}"] = info

        return np.copy(self._obs_buf), self._aggregate_infos(infos, ids)

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """
        在所有环境中执行一步。

        Args:
            actions: 一个动作数组，形状为 (num_envs, ...)。

        Returns:
            (observations, rewards, terminateds, truncateds, infos)
        """
        # 1. 首先重置那些在上一步中被标记为结束的环境
        if np.any(self._autoreset_flags):
            reset_ids = np.where(self._autoreset_flags)[0]
            # 注意：这里的 reset 调用返回的 obs 和 info 我们暂时不需要
            # 因为它们对应的是新 episode 的开始，而我们需要的是上一个 episode 的最终信息
            # gymnasium 的标准是，step 返回的 obs 是新 episode 的第一个 obs
            # 而 info 中包含上一个 episode 的 final_observation
            for env_id in reset_ids:
                obs, info = self.envs[env_id].reset()
                self._obs_buf[env_id] = obs
            self._autoreset_flags[:] = False  # 清空标记

        infos = {}
        # 2. 在每个环境中执行一步
        for i in range(self.num_envs):
            obs, reward, terminated, truncated, info = self.envs[i].step(actions[i])

            self._rewards_buf[i] = reward
            self._terminateds_buf[i] = terminated
            self._truncateds_buf[i] = truncated

            if terminated or truncated:
                # 标记此环境需要在下一次 step 开始时自动重置
                self._autoreset_flags[i] = True
                # 保存最终观测和信息
                info["final_observation"] = obs
                info["final_info"] = info.copy()

            # 无论是否结束，都更新观测缓冲区
            # 如果结束了，这个 obs 是新 episode 的第一个 obs
            # 如果没结束，这个 obs 是当前 episode 的下一个 obs
            self._obs_buf[i] = obs

            infos[f"_{i}"] = info

        aggregated_info = self._aggregate_infos(infos)
        return (
            np.copy(self._obs_buf),
            np.copy(self._rewards_buf),
            np.copy(self._terminateds_buf),
            np.copy(self._truncateds_buf),
            aggregated_info,
        )

    def close(self):
        """关闭所有环境。"""
        for env in self.envs:
            env.close()

    def _aggregate_infos(
        self, infos: dict[str, Any], a_ids: np.ndarray | None = None
    ) -> dict[str, Any]:
        """
        聚合来自多个环境的 info 字典。
        - 将同名键的值合并成一个数组。
        - 处理 `final_observation` 和 `final_info` 等特殊键。
        - a_ids: active_ids, 表示哪些环境的 info 是有效的
        """
        if a_ids is None:
            a_ids = np.arange(self.num_envs)

        aggregated = {}
        all_keys = set()
        for i in a_ids:
            if f"_{i}" in infos:
                all_keys.update(infos[f"_{i}"].keys())

        for key in all_keys:
            # 检查是否所有环境都返回了这个键
            is_present = [
                f"_{i}" in infos and key in infos[f"_{i}"] for i in range(self.num_envs)
            ]

            if all(is_present):
                # 如果所有环境都有这个key，直接堆叠
                values = [infos[f"_{i}"][key] for i in range(self.num_envs)]
                try:
                    aggregated[key] = np.stack(values)
                except (ValueError, TypeError):
                    # 如果无法堆叠（例如，不同形状或类型），则保持为列表
                    aggregated[key] = values
            else:
                # 如果只有部分环境有这个key，创建一个对象数组并用None填充
                values = [
                    infos[f"_{i}"].get(key) if f"_{i}" in infos else None
                    for i in range(self.num_envs)
                ]
                aggregated[key] = np.array(values, dtype=object)

        # 最终的 info 应该只包含那些活跃环境的 `_` 掩码
        aggregated["_"] = np.zeros(self.num_envs, dtype=bool)
        aggregated["_"][a_ids] = True

        return aggregated

    def _create_vectorized_space(self, space: gym.Space, n: int) -> gym.Space:
        """根据单个环境的空间创建向量化空间。"""
        if isinstance(space, gym.spaces.Box):
            return gym.spaces.Box(
                low=np.repeat(np.expand_dims(space.low, 0), n, axis=0),
                high=np.repeat(np.expand_dims(space.high, 0), n, axis=0),
                dtype=space.dtype,  # type: ignore
            )
        elif isinstance(space, gym.spaces.Discrete):
            # 对于离散空间，通常返回一个 Box 来表示多个离散值
            return gym.spaces.Box(
                low=0, high=space.n - 1, shape=(n,), dtype=space.dtype  # type: ignore
            )
        elif isinstance(space, gym.spaces.Dict):
            return gym.spaces.Dict(
                {
                    key: self._create_vectorized_space(sub_space, n)
                    for key, sub_space in space.spaces.items()
                }
            )
        else:
            raise NotImplementedError(f"不支持对 {type(space)} 空间进行自动向量化。")

    @property
    def unwrapped(self):
        return self
