import torch
import torch.nn as nn
import torch.nn.functional as F

class RawCentralizedCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, n_agents, hidden_dim=128):
        super().__init__()
        input_dim = n_agents * (obs_dim + act_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_list, goal_list_unused, act_list):
        inputs = torch.cat(obs_list + act_list, dim=-1)
        return self.net(inputs)
