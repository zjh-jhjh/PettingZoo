import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.goal_encoder import GoalEncoder


class GoalConditionedActor(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = GoalEncoder(obs_dim, goal_dim, latent_dim)

        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围 ∈ [-1, 1]
        )

    def forward(self, obs, goal):
        """
        obs: (B, obs_dim)
        goal: (B, goal_dim)
        return: (B, action_dim)
        """
        z = self.encoder(obs, goal)
        return self.policy_net(z)
