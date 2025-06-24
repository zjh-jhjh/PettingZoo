# modules/goal_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# goal_encoder.py
class GoalEncoder(nn.Module):
    def __init__(self, obs_dim, goal_dim, latent_dim=64):
        super().__init__()
        self.input_dim = obs_dim + goal_dim  # ✅ 加上这一行
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)
        return self.net(x)

