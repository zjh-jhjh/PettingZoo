import torch
import torch.nn as nn

class RawCentralizedCritic(nn.Module):
    def __init__(self, obs_dims, act_dim, hidden_dim=128):
        super().__init__()
        assert isinstance(obs_dims, list), "obs_dims 应该是一个 list，例如 [8, 10, 10]"
        input_dim = sum(obs_dims) + len(obs_dims) * act_dim
        print(f"[RawCritic] Total input dim: {input_dim}")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs_list, act_list):
        x = torch.cat(obs_list + act_list, dim=-1)
        assert x.shape[-1] == self.net[0].in_features, \
            f"[RawCritic] 输入维度 {x.shape[-1]} 和期望 {self.net[0].in_features} 不一致！"
        return self.net(x)

