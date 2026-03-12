import torch
import torch.nn as nn


class RawCentralizedCritic(nn.Module):
    def __init__(self, obs_dims, act_dim, hidden_dim=128):
        """
        Raw 模式的集中式 Critic：
        输入 = concat(all_obs, all_actions)
        obs_dims: list，每个智能体的 obs 维度，例如 [8, 10, 10]
        act_dim: 每个 agent 的动作维度（假设相同）
        """
        super().__init__()
        assert isinstance(obs_dims, list), "obs_dims 应该是一个 list，例如 [8, 10, 10]"

        self.obs_dims = obs_dims
        self.act_dim = act_dim
        self.n_agents = len(obs_dims)

        input_dim = sum(obs_dims) + self.n_agents * act_dim
        print(f"[RawCritic] Total input dim: {input_dim}")

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q 值标量
        )

    def forward(self, obs_list, act_list):
        processed = []

        for t in obs_list + act_list:
            # ✅ 确保所有输入都是 [B, dim] 形式
            if t.dim() == 1:
                t = t.unsqueeze(0)
            processed.append(t)

        x = torch.cat(processed, dim=-1)
        assert x.shape[-1] == self.net[0].in_features, \
            f"[RawCritic] 输入维度 {x.shape[-1]} 和期望 {self.net[0].in_features} 不一致！"
        return self.net(x)
