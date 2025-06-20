import torch
import torch.nn as nn
import torch.nn.functional as F

class RawActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Sigmoid()  # 将 Tanh 改为 Sigmoid，输出范围变为 [0,1]
        )

    def forward(self, obs, goal=None):  # 保持接口一致
        return self.net(obs)
