import torch
import torch.nn as nn
from modules.goal_encoder import GoalEncoder


class CentralizedCritic(nn.Module):
    def __init__(self, obs_dims, goal_dims, act_dim, latent_dim=64, hidden_dim=128):
        """
        Encoder 模式下的集中式 Critic:
        输入 = concat(每个 agent 的 latent 表示, 所有动作)
        obs_dims: list，每个 agent 的 obs_dim
        goal_dims: list，每个 agent 的 goal_dim
        act_dim: 每个 agent 的动作维度（假设相同）
        """
        super().__init__()
        self.n_agents = len(obs_dims)
        self.obs_dims = obs_dims
        self.goal_dims = goal_dims
        self.act_dim = act_dim

        # 为每个 agent 定义一个 GoalEncoder
        self.encoders = nn.ModuleList([
            GoalEncoder(obs_dims[i], goal_dims[i], latent_dim)
            for i in range(self.n_agents)
        ])

        # Critic 输入：所有 agent latent + 所有 agent action
        input_dim = self.n_agents * latent_dim + self.n_agents * act_dim
        print(f"[CentralizedCritic] Total input dim: {input_dim}")

        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Q 值标量
        )

    def forward(self, obs_list, goal_list, act_list):
        z_list = []
        for enc, o, g in zip(self.encoders, obs_list, goal_list):
            # 保证 o 和 g 都是 2D
            if o.dim() == 1:
                o = o.unsqueeze(0)
            if g.dim() == 1:
                g = g.unsqueeze(0)
            z_list.append(enc(o, g))

        # 统一 act_list 维度
        act_inputs = []
        for a in act_list:
            if a.dim() == 1:
                a = a.unsqueeze(0)
            act_inputs.append(a)

        # 拼接
        q_input = torch.cat(z_list + act_inputs, dim=-1)
        return self.q_net(q_input)
