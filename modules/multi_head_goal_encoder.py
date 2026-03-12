import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadGoalEncoder(nn.Module):
    """
    多头注意力机制的 GoalEncoder
    输入:
        obs: (B, obs_dim)
        goal: (B, goal_dim)
    输出:
        z: (B, latent_dim)
    """

    def __init__(self, obs_dim, goal_dim, latent_dim=64, num_heads=4):
        super().__init__()
        assert latent_dim % num_heads == 0, "latent_dim 必须能被 num_heads 整除"

        self.obs_dim = obs_dim
        self.goal_dim = goal_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads

        # 线性映射到 Q, K, V 空间
        self.obs_proj = nn.Linear(obs_dim, latent_dim)
        self.goal_proj_q = nn.Linear(goal_dim, latent_dim)
        self.goal_proj_k = nn.Linear(goal_dim, latent_dim)
        self.goal_proj_v = nn.Linear(goal_dim, latent_dim)

        # 融合层
        self.fc_out = nn.Sequential(
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
        B = obs.size(0)

        # (B, latent_dim)
        obs_feat = self.obs_proj(obs)
        q = self.goal_proj_q(goal)
        k = self.goal_proj_k(goal)
        v = self.goal_proj_v(goal)

        # reshape for multi-head: (B, num_heads, head_dim)
        obs_feat = obs_feat.view(B, self.num_heads, self.head_dim)
        q = q.view(B, self.num_heads, self.head_dim)
        k = k.view(B, self.num_heads, self.head_dim)
        v = v.view(B, self.num_heads, self.head_dim)

        # 注意力分数: (B, num_heads, head_dim)
        attn_scores = torch.sum(q * k, dim=-1, keepdim=True) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, num_heads, 1)

        # 加权组合: (B, num_heads, head_dim)
        attended = attn_weights * obs_feat + (1 - attn_weights) * v

        # 拼接 heads -> (B, latent_dim)
        attended = attended.view(B, self.latent_dim)

        # 输出层
        z = self.fc_out(attended)
        return z
