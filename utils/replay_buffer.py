import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, buffer_size, obs_dim, goal_dim, act_dim, n_agents, device):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim  # 单个智能体的观测维度
        self.goal_dim = goal_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.device = device

        # 初始化缓冲区
        self.obs = np.zeros((buffer_size, obs_dim))
        self.next_obs = np.zeros((buffer_size, obs_dim))
        self.goals = np.zeros((buffer_size, goal_dim))
        self.next_goals = np.zeros((buffer_size, goal_dim))
        self.actions = np.zeros((buffer_size, act_dim))
        self.rewards = np.zeros((buffer_size, 1))
        self.dones = np.zeros((buffer_size, 1))
        self.ptr = 0
        self.size = 0

    def add(self, obs, act, next_obs, next_goals, reward, done, info):
        """
        添加数据到缓冲区
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = act
        self.next_obs[self.ptr] = next_obs
        self.next_goals[self.ptr] = next_goals
        self.rewards[self.ptr] = reward if reward is not None else 0.0
        self.dones[self.ptr] = done if done is not None else 0.0
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        """
        :return: dict of tensors with keys: obs, obs_next, goals, goals_next, actions, rewards, dones
        Each of shape (B, dim)
        """
        idx = np.random.choice(self.size, batch_size, replace=False)

        def to_tensor(x):
            return torch.tensor(x[idx], dtype=torch.float32, device=self.device)

        batch = {
            "obs": to_tensor(self.obs),           # (B, obs_dim)
            "obs_next": to_tensor(self.next_obs),
            "goals": to_tensor(self.goals),
            "goals_next": to_tensor(self.next_goals),
            "actions": to_tensor(self.actions),
            "rewards": to_tensor(self.rewards),   # (B, 1)
            "dones": to_tensor(self.dones)        # (B, 1)
        }
        return batch

    def __len__(self):
        return self.size
