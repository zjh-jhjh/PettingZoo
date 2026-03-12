import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.goal_encoder import GoalEncoder


class GoalConditionedActor(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, latent_dim=64, hidden_dim=128, use_encoder=True):
        super().__init__()
        self.use_encoder = use_encoder

        if self.use_encoder:
            # encoder 模式：obs 与 goal 分开送入 GoalEncoder
            self.encoder = GoalEncoder(obs_dim, goal_dim, latent_dim)
            input_dim = latent_dim
        else:
            # raw 模式：obs 已经包含 goal_rel，所以直接作为输入
            self.encoder = None
            input_dim = obs_dim

        self.policy_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # ✅ [-1, 1] 范围更合理，替换掉 Sigmoid
        )

    def forward(self, obs, goal=None):
        if self.use_encoder:
            # obs 和 goal 分开输入，交给 GoalEncoder
            assert goal is not None, "❌ Encoder 模式下必须提供 goal"
            z = self.encoder(obs, goal)
        else:
            # raw 模式下 obs 已经包含 goal_rel，不再单独传 goal
            z = obs

        return self.policy_net(z)
