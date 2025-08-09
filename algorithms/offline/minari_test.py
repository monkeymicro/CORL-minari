import minari
import numpy as np
import torch

# 假设你已有的 ReplayBuffer 类如下（简化）：
class SimpleReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, size: int,):
        self._states = np.zeros([size, obs_dim], dtype=np.float32)
        self._actions = np.zeros([size, action_dim], dtype=np.float32)
        self._rewards = np.zeros([size], dtype=np.float32)
        self._next_states = np.zeros([size, obs_dim], dtype=np.float32)
        self._dones = np.zeros([size], dtype=np.float32)
        self.max_size = size
        self.ptr, self.size, = 0, 0
    def store(self, obs: np.ndarray,
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: float,):

        self._states[self.ptr] = obs
        self._next_states[self.ptr] = next_obs
        self._actions[self.ptr] = act
        self._rewards[self.ptr] = rew
        self._dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        # print(f"Loaded {len(data['observations'])} transitions.")

# 选择 Minari 数据集（需提前安装）
dataset_name = 'mujoco/hopper/expert-v0'
dataset = minari.load_dataset(dataset_name, download=True)
print(dataset.total_steps)
# 获取维度信息
env_spec = dataset.env_spec
state_dim = dataset.observation_space.shape[0]
action_dim = dataset.action_space.shape[0]
print('state dim is ', state_dim)

# 创建你的 ReplayBuffer
buffer = SimpleReplayBuffer(state_dim, action_dim, dataset.total_steps)

# 使用 iterate_episodes 来提取数据
for ep in dataset.iterate_episodes():
    for i in range(ep.observations.shape[0] - 1):
        obs = ep.observations[i]
        # print(obs)
        next_obs = ep.observations[i+1]
        action = ep.actions[i]
        reward = ep.rewards[i]
        done = (ep.terminations | ep.truncations)[i].astype(np.float32)

        buffer.store(obs, action, reward, next_obs, done)

print("Replay buffer filled.")
