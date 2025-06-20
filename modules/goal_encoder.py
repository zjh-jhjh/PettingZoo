# modules/goal_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GoalEncoder(nn.Module):
    def __init__(self, obs_dim, goal_dim, hidden_dim):
        super(GoalEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, obs, goal):
        x = torch.cat([obs, goal], dim=-1)
        return self.net(x)
