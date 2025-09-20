import gymnasium as gym
import numpy as np
import time

from envrunner import EnvExecutor


# --- 1. 定义一个伪策略 (Fake Policy) ---
class FakeStochasticPolicy:
    """
    一个简单的伪策略，它根据动作空间进行随机采样。
    这模拟了一个在 GPU 上运行的神经网络策略，但它只在 CPU 上运行。
    """

    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    def get_actions(self, observations: np.ndarray) -> np.ndarray:
        """
        接收一个批量的观测值，返回一个批量的动作。

        Args:
            observations: 一个 NumPy 数组，形状为 (num_envs, ...)，代表一批观测值。

        Returns:
            一个 NumPy 数组，形状为 (num_envs,)，代表为每个环境选择的动作。
        """
        num_envs = observations.shape[0]

        # 模拟策略网络进行推理所需的时间
        # time.sleep(0.002)

        # 对于离散动作空间，为每个环境随机采样一个动作
        if isinstance(self.action_space, gym.spaces.Discrete):
            # np.random.randint 比循环调用 action_space.sample() 更高效
            return np.random.randint(0, self.action_space.n, size=(num_envs,))

        # 对于连续动作空间
        elif isinstance(self.action_space, gym.spaces.Box):
            return np.array([self.action_space.sample() for _ in range(num_envs)])

        else:
            raise NotImplementedError(
                f"不支持的动作空间类型: {type(self.action_space)}"
            )


# --- 2. 设置参数和初始化 ---
def main():
    # --- 参数配置 ---
    ENV_ID = "CartPole-v1"
    TOTAL_ENVS = 1024  # 要运行的环境总数
    NUM_WORKERS = 32  # 使用的 CPU 核心/进程数

    # 模拟的总步数
    TOTAL_TIMESTEPS = 10_000

    # 确保环境总数可以被 worker 数量整除
    if TOTAL_ENVS % NUM_WORKERS != 0:
        raise ValueError("环境总数必须能被 worker 数量整除！")

    print("=" * 40)
    print("高性能环境执行器 - 示例")
    print(f"环境: {ENV_ID}")
    print(f"总环境数: {TOTAL_ENVS}")
    print(f"工作进程数: {NUM_WORKERS}")
    print(f"总步数: {TOTAL_TIMESTEPS}")
    print("=" * 40)

    # --- 初始化 ---
    # 1. 创建环境构造函数列表
    env_fns = [lambda: gym.make(ENV_ID) for _ in range(TOTAL_ENVS)]

    # 2. 初始化高性能环境执行器
    # 我们将它放在一个 try...finally 块中，以确保 close() 总能被调用
    executor = None
    try:
        executor = EnvExecutor(env_fns, num_workers=NUM_WORKERS)

        # 3. 初始化伪策略
        policy = FakeStochasticPolicy(executor.action_space)

        # --- 3. 强化学习主循环 ---
        print("\n开始数据采样循环...")
        start_time = time.time()

        # 重置所有环境，获取初始观测值
        observations, infos = executor.reset(seed=42)

        episode_returns = []
        episode_lengths = []

        for step in range(TOTAL_TIMESTEPS):
            # 1. 策略根据当前观测值选择动作
            # 这是“超大的帧”一次性提交给策略
            actions = policy.get_actions(observations)

            # 2. 执行器在所有环境中执行一步
            next_obs, rewards, terminateds, truncateds, infos = executor.step(actions)

            # 3. 处理返回的信息
            # `infos` 字典包含了最终的观测和回报等信息
            if "final_info" in infos:
                # 过滤出那些真正结束了的环境
                done_envs_info = [
                    info for info in infos["final_info"] if info is not None
                ]
                for info in done_envs_info:
                    if "episode" in info:
                        ret = info["episode"]["r"][0]
                        length = info["episode"]["l"][0]
                        episode_returns.append(ret)
                        episode_lengths.append(length)
                        print(
                            f"  [步数 {step+1}/{TOTAL_TIMESTEPS}] Episode 结束. 回报: {ret}, 长度: {length}"
                        )

            # 4. 更新观测值以进行下一步
            observations = next_obs

            # 5. 打印进度
            if (step + 1) % 100 == 0:
                progress = (step + 1) / TOTAL_TIMESTEPS * 10
                print(f"进度: {step+1}/{TOTAL_TIMESTEPS} ({progress:.2f}%)")

        end_time = time.time()
        print("\n...采样循环结束。")

        # --- 4. 性能和结果统计 ---
        total_time = end_time - start_time
        fps = TOTAL_TIMESTEPS / total_time

        print("\n--- 统计结果 ---")
        print(f"总耗时: {total_time:.2f} 秒")
        print(f"帧率 (FPS): {fps:.0f} 帧/秒")
        if episode_returns:
            print(f"平均 Episode 回报: {np.mean(episode_returns):.2f}")
            print(f"平均 Episode 长度: {np.mean(episode_lengths):.2f}")
        print("-" * 18)

    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        # --- 5. 清理资源 ---
        if executor:
            print("\n正在关闭执行器...")
            executor.close()
            print("执行器已关闭。")


if __name__ == "__main__":
    main()
