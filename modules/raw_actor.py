import torch
import torch.nn as nn


class RawActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        """
        非目标导向 Actor，不使用 goal，只基于 obs 输入。
        """
        super().__init__()

        self.obs_dim = obs_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Sigmoid()  # 输出范围 [0 , 1]，与主策略保持一致
        )

    def forward(self, obs: torch.Tensor, goal=None) -> torch.Tensor:
        """
        obs: shape (B, obs_dim)
        return: shape (B, act_dim)
        goal: 无实际用途，仅为了接口兼容 GoalConditionedActor
        """
        assert obs.shape[-1] == self.obs_dim, \
            f"RawActor: obs 输入维度应为 {self.obs_dim}，实际为 {obs.shape[-1]}"
        return self.net(obs)
