from typing import Callable, Sequence
import uuid
import traceback
import threading
import multiprocessing as mp
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Barrier

import numpy as np
import gymnasium as gym

from .sync_vector_env import SyncSubVectorEnv


# --- Shared Memory Helper Functions ---
def _create_shared_memory(
    name: str, shape: tuple[int, ...], dtype: np.dtype
) -> tuple[SharedMemory, np.ndarray]:
    """创建一个共享内存块并返回其对象和对应的 numpy 数组视图。"""
    size = int(np.prod(shape) * np.dtype(dtype).itemsize)
    shm = SharedMemory(name=name, create=True, size=size)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, array


def _get_shared_memory(
    name: str, shape: tuple[int, ...], dtype: np.dtype
) -> tuple[SharedMemory, np.ndarray]:
    """连接到一个已存在的共享内存块。"""
    shm = SharedMemory(name=name, create=False)
    array = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, array


def worker(
    worker_id: int,
    env_fns: Sequence[Callable[[], gym.Env]],
    main_pipe: Connection,
    worker_pipe: Connection,
    barrier: Barrier,
    sm_meta: dict[str, tuple[str, tuple, np.dtype]],
    timeout: int = 10,
):
    """子进程的工作函数。"""
    # 1. 初始化
    main_pipe.close()  # 在子进程中关闭主进程的管道端

    vec_env = None
    shm_objects = {}
    try:
        # 创建环境
        vec_env = SyncSubVectorEnv(env_fns)
        envs_per_worker = vec_env.num_envs

        # 连接共享内存
        shm_arrays = {}
        for key, (name, shape, dtype) in sm_meta.items():
            shm, arr = _get_shared_memory(name, shape, dtype)
            shm_objects[key] = shm
            shm_arrays[key] = arr

        # 计算此 worker 负责的内存切片
        start = worker_id * envs_per_worker
        end = start + envs_per_worker
        obs_slice = shm_arrays["obs"][start:end]
        rew_slice = shm_arrays["rew"][start:end]
        term_slice = shm_arrays["term"][start:end]
        trunc_slice = shm_arrays["trunc"][start:end]

        # 发送 "ready" 信号
        worker_pipe.send("ready")

        # 2. 主循环
        while True:
            try:
                cmd, data = worker_pipe.recv()
            except (EOFError, ConnectionResetError):
                break  # 管道被主进程关闭

            if cmd == "reset":
                obs, info = vec_env.reset(**data)
                obs_slice[:] = obs
                worker_pipe.send(info)
                try:
                    barrier.wait(timeout=timeout)
                except (threading.BrokenBarrierError, TimeoutError):
                    break

            elif cmd == "step":
                obs, rewards, term, trunc, info = vec_env.step(data)
                obs_slice[:] = obs
                rew_slice[:] = rewards
                term_slice[:] = term
                trunc_slice[:] = trunc
                worker_pipe.send(info)
                try:
                    barrier.wait(timeout=timeout)
                except (threading.BrokenBarrierError, TimeoutError):
                    break

            elif cmd == "close":
                break

            else:
                worker_pipe.send({"error": f"未知指令: {cmd}"})
                break

    except Exception as e:
        # 捕获任何异常并通过管道发送给主进程
        tb = traceback.format_exc()
        worker_pipe.send({"error": e, "traceback": tb})

    finally:
        # 3. 清理资源
        if vec_env:
            vec_env.close()
        for shm in shm_objects.values():
            shm.close()
        worker_pipe.close()


class EnvExecutor:
    def __init__(
        self, env_fns: list[Callable[[], gym.Env]], num_workers: int, timeout: int = 10
    ):
        self.timeout = timeout
        self.closed = True  # 预设为已关闭，防止异常时误操作

        if num_workers <= 0:
            raise ValueError("num_workers 必须为正整数。")

        self.num_envs = len(env_fns)
        self.num_workers = num_workers

        # 1. 分配环境给每个 worker
        self.envs_per_worker = self.num_envs // self.num_workers
        if self.num_envs % self.num_workers != 0:
            raise ValueError("环境总数必须能被 worker 数量整除，以简化设计。")

        # 手动分割环境函数列表
        self.worker_env_fns = []
        for i in range(self.num_workers):
            start_idx = i * self.envs_per_worker
            end_idx = start_idx + self.envs_per_worker
            self.worker_env_fns.append(env_fns[start_idx:end_idx])

        # 2. 推断空间 (从单个环境实例)
        temp_env = env_fns[0]()
        self.single_observation_space = temp_env.observation_space
        self.single_action_space = temp_env.action_space
        temp_env.close()

        # 向量化环境的属性
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space

        # 3. 创建共享内存
        unique_id = str(uuid.uuid4())  # 确保共享内存名称唯一
        self.sm_meta = {
            "obs": (
                f"hpe_obs_{unique_id}",
                (self.num_envs,) + self.observation_space.shape,  # type: ignore
                self.observation_space.dtype,
            ),
            "rew": (f"hpe_rew_{unique_id}", (self.num_envs,), np.float32),
            "term": (f"hpe_term_{unique_id}", (self.num_envs,), np.bool_),
            "trunc": (f"hpe_trunc_{unique_id}", (self.num_envs,), np.bool_),
        }

        self._shm_objects = {}
        self._shm_arrays = {}
        try:
            for key, (name, shape, dtype) in self.sm_meta.items():
                shm, arr = _create_shared_memory(name, shape, dtype)
                self._shm_objects[key] = shm
                self._shm_arrays[key] = arr
        except Exception as e:
            self.close()  # 如果创建失败，清理已创建的共享内存
            raise e

        # 4. 创建通信和同步工具
        self.pipes = [mp.Pipe() for _ in range(self.num_workers)]
        # Barrier 等待所有 worker 和主进程
        self.barrier = mp.Barrier(self.num_workers + 1)

        # 5. 创建并启动子进程
        self.workers = []
        for i in range(self.num_workers):
            main_pipe, worker_pipe = self.pipes[i]
            proc = mp.Process(
                target=worker,
                args=(
                    i,
                    self.worker_env_fns[i],
                    main_pipe,
                    worker_pipe,
                    self.barrier,
                    self.sm_meta,
                ),
                daemon=True,  # 设置为守护进程，主进程退出时自动终止
            )
            self.workers.append(proc)
            proc.start()
            worker_pipe.close()  # 在主进程中关闭 worker 端的 pipe

        # 等待所有 worker 完成初始化并发出 "ready" 或错误信号
        for i in range(self.num_workers):
            try:
                status = self.pipes[i][0].recv()
            except (EOFError, ConnectionResetError):
                # 如果管道在收到任何消息前就关闭了，说明 worker 提前崩溃了
                self.close()
                raise RuntimeError(f"Worker {i} 在初始化期间意外退出。")

            # 检查 status 是否为错误字典
            if isinstance(status, dict) and "error" in status:
                # 如果是，关闭所有资源并抛出包含详细信息的异常
                self.close()
                raise RuntimeError(
                    f"Worker {i} 初始化失败: {status['error']}\n{status['traceback']}"
                )

            if status != "ready":
                # 对于其他非预期的状态
                self.close()
                raise RuntimeError(f"Worker {i} 返回了未知状态: {status}")

        self.closed = False

    def _wait_on_barrier(self):
        """带超时的屏障等待的辅助函数"""
        try:
            self.barrier.wait(timeout=self.timeout)
        except (threading.BrokenBarrierError, TimeoutError):
            self.close()  # 立即清理
            raise RuntimeError(
                "一个或多个工作进程未能同步（可能已崩溃或死锁）。执行器已关闭。"
            )

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
        ids: list[int] | None = None,
    ) -> tuple[np.ndarray, dict]:
        if self.closed:
            raise gym.error.ClosedEnvironmentError("执行器已关闭。")

        # TODO: 处理复杂的分布式 ids
        if ids is not None:
            raise NotImplementedError("部分重置 (ids) 在当前版本中尚未实现。")

        # 1. 向所有 worker 发送 reset 指令
        for i in range(self.num_workers):
            # 为每个 worker 计算种子
            worker_seed = seed + i * self.envs_per_worker if seed is not None else None
            self.pipes[i][0].send(("reset", {"seed": worker_seed, "options": options}))

        # 1. 收集 info
        all_infos = []
        worker_errors = []
        for i in range(self.num_workers):
            received_data = self.pipes[i][0].recv()
            if isinstance(received_data, dict) and "error" in received_data:
                worker_errors.append(received_data)
            else:
                all_infos.append(received_data)

        # 2. 优先处理 worker 错误
        if worker_errors:
            # 获取第一个遇到的错误并抛出
            first_error = worker_errors[0]
            self.close()
            raise RuntimeError(
                f"子进程发生错误: {first_error['error']}\n{first_error['traceback']}"
            )

        # 3. 在屏障处同步
        self._wait_on_barrier()

        # 4. 从共享内存读取观测值并聚合 info
        obs = np.copy(self._shm_arrays["obs"])
        aggregated_infos = self._aggregate_worker_infos(all_infos)

        return obs, aggregated_infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        if self.closed:
            raise gym.error.ClosedEnvironmentError("执行器已关闭。")

        # 1. 将动作切分并发送给每个 worker
        action_slices = np.split(actions, self.num_workers)
        for i in range(self.num_workers):
            self.pipes[i][0].send(("step", action_slices[i]))

        # 1. 收集 info
        all_infos = []
        worker_errors = []
        for i in range(self.num_workers):
            received_data = self.pipes[i][0].recv()
            if isinstance(received_data, dict) and "error" in received_data:
                worker_errors.append(received_data)
            else:
                all_infos.append(received_data)

        # 2. 优先处理 worker 错误
        if worker_errors:
            # 获取第一个遇到的错误并抛出
            first_error = worker_errors[0]
            self.close()
            raise RuntimeError(
                f"子进程发生错误: {first_error['error']}\n{first_error['traceback']}"
            )

        # 3. 在屏障处同步
        self._wait_on_barrier()

        # 4. 从共享内存读取结果并聚合 info
        obs = np.copy(self._shm_arrays["obs"])
        rewards = np.copy(self._shm_arrays["rew"])
        terminateds = np.copy(self._shm_arrays["term"])
        truncateds = np.copy(self._shm_arrays["trunc"])

        aggregated_infos = self._aggregate_worker_infos(all_infos)

        return obs, rewards, terminateds, truncateds, aggregated_infos

    def close(self):
        if hasattr(self, "closed") and self.closed:
            return

        if hasattr(self, "pipes"):
            for pipe, _ in self.pipes:
                try:
                    pipe.send(("close", None))
                except (BrokenPipeError, ConnectionResetError):
                    pass

        if hasattr(self, "workers"):
            for p in self.workers:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()

        # 清理共享内存（即使对象只创建了一部分）
        if hasattr(self, "_shm_objects"):
            for shm in self._shm_objects.values():
                shm.close()
                try:
                    shm.unlink()
                except FileNotFoundError:
                    pass

        if hasattr(self, "pipes"):
            for pipe, _ in self.pipes:
                pipe.close()

        self.closed = True

    def _aggregate_worker_infos(self, worker_infos: list[dict]) -> dict:
        """将来自不同 worker 的 info 字典聚合成一个。"""
        if not worker_infos:
            return {}

        aggregated_info = {}

        # 1. 收集所有 keys
        all_keys = set()
        for info in worker_infos:
            all_keys.update(info.keys())

        # 2. 聚合
        for key in all_keys:
            # 提取所有 worker 的数据片段
            # 如果某个 worker 的 info 中没有这个 key，我们用一个占位符来表示
            # 这个占位符将是一个包含 None 的列表/数组
            slices_to_concat = []
            for i, info in enumerate(worker_infos):
                num_envs_worker = len(self.worker_env_fns[i])
                # 安全地获取数据，如果不存在则创建一个由 None 组成的占位符
                slice_data = info.get(key, np.full(num_envs_worker, None, dtype=object))
                slices_to_concat.append(slice_data)

            # 使用 np.concatenate 将所有片段拼接起来
            # np.concatenate 可以处理不同 worker 的数组
            try:
                aggregated_info[key] = np.concatenate(slices_to_concat, axis=0)
            except ValueError:
                # 如果因为类型不匹配等原因无法拼接，退化为普通列表拼接
                final_list = []
                for s in slices_to_concat:
                    final_list.extend(s)
                aggregated_info[key] = np.array(final_list, dtype=object)

        return aggregated_info

    def __del__(self):
        self.close()
