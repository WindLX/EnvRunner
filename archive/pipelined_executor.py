# pipelined_executor.py

import threading
import time
import numpy as np
import torch

from ..src.envrunner.env_executor import EnvExecutor


class PipelinedExecutor:
    def __init__(self, env_fns, num_workers: int, batch_size: int, policy):
        if len(env_fns) != batch_size:
            raise ValueError("env_fns 长度必须等于 batch_size")

        self.policy = policy
        self.batch_size = batch_size
        self.closed = False

        # --- 1. 创建两个独立的执行器 ---
        # 我们将总的环境列表分成两半
        split_index = batch_size // 2
        self.executors = [
            EnvExecutor(env_fns[:split_index], num_workers=num_workers // 2),
            EnvExecutor(
                env_fns[split_index:], num_workers=num_workers - num_workers // 2
            ),
        ]

        # --- 2. 准备数据缓冲区 ---
        # 我们直接使用执行器内部的共享内存数组作为缓冲区
        self.obs_buffers = [
            torch.from_numpy(self.executors[0]._shm_arrays["obs"]).pin_memory(),
            torch.from_numpy(self.executors[1]._shm_arrays["obs"]).pin_memory(),
        ]

        # --- 3. 创建同步事件 ---
        # buffer_ready_for_gpu[i] 事件表示第 i 个缓冲区已装满数据，GPU 可以使用
        self.buffer_ready_for_gpu = [threading.Event(), threading.Event()]
        # buffer_free_for_cpu[i] 事件表示第 i 个缓冲区已被 GPU 用完，CPU 可以重新填充
        self.buffer_free_for_cpu = [threading.Event(), threading.Event()]

        # 初始状态：两个缓冲区都可供 CPU 使用
        self.buffer_free_for_cpu[0].set()
        self.buffer_free_for_cpu[1].set()

        # --- 4. 启动 CPU 采样线程 ---
        self.cpu_thread = threading.Thread(target=self._cpu_sample_loop, daemon=True)
        self.cpu_thread.start()

    def _cpu_sample_loop(self):
        """
        这个函数在一个独立的线程中运行，只负责采集数据。
        """
        # 初始重置
        self.executors[0].reset(seed=42)
        self.executors[1].reset(seed=42 + self.executors[0].num_envs)

        current_buffer_idx = 0
        while not self.closed:
            # 等待当前缓冲区变为空闲
            self.buffer_free_for_cpu[current_buffer_idx].wait()
            self.buffer_free_for_cpu[current_buffer_idx].clear()

            # --- 核心采样逻辑 ---
            # 1. 从对应的共享内存中获取最新的观测值 (不需要拷贝!)
            obs_np = self.executors[current_buffer_idx]._shm_arrays["obs"]

            # 2. GPU 推理获取动作
            actions = self.policy.get_actions(obs_np)

            # 3. 执行一步
            # 这是一个阻塞调用，会等待这个执行器完成采样
            self.executors[current_buffer_idx].step(actions)
            # 采样完成后，缓冲区的数据已经自动更新在共享内存中了

            # 通知 GPU，这个缓冲区的数据已经准备好了
            self.buffer_ready_for_gpu[current_buffer_idx].set()

            # 切换到下一个缓冲区
            current_buffer_idx = 1 - current_buffer_idx

    def __iter__(self):
        """使这个类成为一个迭代器，可以轻松地在训练循环中使用。"""
        self.current_gpu_buffer_idx = 0
        return self

    def __next__(self):
        """
        这个方法在主训练线程（GPU线程）中被调用。
        它会阻塞，直到下一个数据批次准备就绪。
        """
        if self.closed:
            raise StopIteration

        # 等待当前 GPU 缓冲区的数据被 CPU 准备好
        self.buffer_ready_for_gpu[self.current_gpu_buffer_idx].wait()
        self.buffer_ready_for_gpu[self.current_gpu_buffer_idx].clear()

        # 获取数据 (直接从固定内存的 torch tensor 获取，零拷贝)
        obs_tensor = self.obs_buffers[self.current_gpu_buffer_idx]

        # ... 在现实中，你还会从执行器中获取 reward, done 等信息
        # reward_np = self.executors[self.current_gpu_buffer_idx]._shm_arrays['rew']
        # ...

        # 通知 CPU，这个缓冲区已经被我们“消费”了，可以再次用于采样
        self.buffer_free_for_cpu[self.current_gpu_buffer_idx].set()

        # 切换到下一个缓冲区
        self.current_gpu_buffer_idx = 1 - self.current_gpu_buffer_idx

        # 返回数据给训练循环
        # 注意：这里只返回了 obs，实际应用中会返回一个包含所有信息的字典
        return {"obs": obs_tensor.to("cuda:0", non_blocking=True)}

    def close(self):
        if not self.closed:
            self.closed = True
            # 简单地通过设置标志位来停止循环
            # 可能需要更复杂的机制来确保线程安全退出
            print("Closing pipelined executor...")
            self.cpu_thread.join(timeout=5)
            self.executors[0].close()
            self.executors[1].close()
            print("Pipelined executor closed.")
