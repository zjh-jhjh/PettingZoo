import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.goal_encoder import GoalEncoder

class CentralizedCritic(nn.Module):
    def __init__(self, obs_dims, goal_dims, act_dim, latent_dim=64, hidden_dim=128):
        """
        obs_dims: list of obs_dim per agent
        goal_dims: list of goal_dim per agent
        """
        super().__init__()
        self.n_agents = len(obs_dims)
        self.encoders = nn.ModuleList([
            GoalEncoder(obs_dims[i], goal_dims[i], latent_dim) for i in range(self.n_agents)
        ])

        input_dim = self.n_agents * (latent_dim + act_dim)
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_list, goal_list, act_list):
        z_list = [enc(o, g) for enc, o, g in zip(self.encoders, obs_list, goal_list)]
        q_input = torch.cat(z_list + act_list, dim=-1)
        return self.q_net(q_input)
