# modules/goal_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class GoalEncoder(nn.Module):
    """
    单头注意力机制的 GoalEncoder
    输入:
        obs: (B, obs_dim)
        goal: (B, goal_dim)
    输出:
        z: (B, latent_dim)
    """

    def __init__(self, obs_dim, goal_dim, latent_dim=64):
        super().__init__()
        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim
        print(f"🧩 GoalEncoder init: obs_dim={obs_dim}, goal_dim={goal_dim}")
        # 将 obs 和 goal 映射到相同隐空间
        self.obs_proj = nn.Linear(obs_dim, latent_dim)
        self.goal_proj = nn.Linear(goal_dim, latent_dim)

        # 注意力打分函数
        self.attn_score = nn.Linear(latent_dim, 1)

        # 输出层
        self.output_net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU()
        )

    def forward(self, obs, goal):
        """
        obs: (B, obs_dim)
        goal: (B, goal_dim)
        """
        # 投影到隐空间
        obs_feat = self.obs_proj(obs)      # (B, latent_dim)
        goal_feat = self.goal_proj(goal)   # (B, latent_dim)

        # 计算注意力权重（单头）
        attn_input = torch.tanh(obs_feat + goal_feat)  # (B, latent_dim)
        score = self.attn_score(attn_input)            # (B, 1)
        weight = torch.sigmoid(score)                  # (B, 1) ∈ (0,1)

        # 加权融合 obs_feat
        attended = weight * obs_feat + (1 - weight) * goal_feat  # (B, latent_dim)

        # 输出表示 f_ω(o, g)
        z = self.output_net(attended)
        return z
