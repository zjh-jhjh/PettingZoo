import torch
import torch.nn as nn

class RawCentralizedCritic(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_list, act_list):
        inputs = torch.cat([torch.cat([obs, act], dim=-1) for obs, act in zip(obs_list, act_list)], dim=-1)
        return self.net(inputs)


