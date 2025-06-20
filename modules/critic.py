import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.goal_encoder import GoalEncoder

class CentralizedCritic(nn.Module):
    def __init__(self, obs_dim, goal_dim, act_dim, n_agents, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.n_agents = n_agents
        self.encoders = nn.ModuleList([
            GoalEncoder(obs_dim, goal_dim, latent_dim) for _ in range(n_agents)
        ])

        input_dim = n_agents * (latent_dim + act_dim)
        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出 Q 值（单个）
        )

    def forward(self, obs_list, goal_list, act_list):
        """
        obs_list: List[Tensor], 每个 shape (B, obs_dim)
        goal_list: List[Tensor], 每个 shape (B, goal_dim)
        act_list: List[Tensor], 每个 shape (B, act_dim)
        return: Q 值 tensor of shape (B, 1)
        """
        z_list = [enc(o, g) for enc, o, g in zip(self.encoders, obs_list, goal_list)]
        inputs = torch.cat(z_list + act_list, dim=-1)  # (B, total_input_dim)
        return self.q_net(inputs)
