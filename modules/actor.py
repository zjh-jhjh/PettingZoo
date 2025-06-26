import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.goal_encoder import GoalEncoder

# encoder模式
class GoalConditionedActor(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, latent_dim=64, hidden_dim=128):
        super().__init__()
        self.encoder = GoalEncoder(obs_dim, goal_dim, latent_dim)

        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # 输出范围 ∈ [-1, 1]
        )

    def forward(self, obs, goal):
        assert obs.shape[-1] + goal.shape[-1] == self.encoder.input_dim, \
            f"❌ 维度不匹配：obs {obs.shape}, goal {goal.shape}, encoder expects {self.encoder.input_dim}"
        z = self.encoder(obs, goal)
        return self.policy_net(z)

