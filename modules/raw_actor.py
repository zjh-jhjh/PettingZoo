import torch
import torch.nn as nn

import torch
import torch.nn as nn


class RawActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 128):
        """
        非目标导向 Actor（Raw 模式）
        输入：obs 已包含 goal_rel（在 scenario.observation 里拼接好的）
        """
        super().__init__()
        self.obs_dim = obs_dim

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh()  # ✅ 输出范围 [-1, 1]，适配 MPE 动作空间
        )

    def forward(self, obs: torch.Tensor, goal=None) -> torch.Tensor:
        """
        obs: shape (B, obs_dim)，包含了 goal_rel（raw 模式）
        goal: 仅为了接口兼容，实际不使用
        return: shape (B, act_dim)，取值范围 [-1,1]
        """
        assert obs.shape[-1] == self.obs_dim, \
            f"RawActor: obs 输入维度应为 {self.obs_dim}，实际为 {obs.shape[-1]}"
        return self.net(obs)
