import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

"""
| 模块               | 输入                    | 输出                        |
| ---------------- | --------------------- | ------------------------- |
| `forward`        | obs + act             | \$D(s,a)\$ ∈ \[0,1]       |
| `compute_reward` | obs + act             | \$r(s,a) = -\log(1 - D)\$ |
| `train_step`     | expert vs agent batch | loss 值（float）             |
"""
def build_discriminator(obs_dim, act_dim, hidden_dim=128):
    """构建一个简单的 MLP 判别器"""
    return nn.Sequential(
        nn.Linear(obs_dim + act_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid()  # 输出 [0, 1] 之间的概率
    )

def collect_trajectory_data(env, policy_fn):
    """收集一条轨迹数据"""
    obs_dict, _ = env.reset()
    done = False
    obs_batch = []
    act_batch = []
    
    while not done:
        # 获取策略动作
        actions = policy_fn(obs_dict)
        
        # 存储当前状态和动作
        obs_batch.append({agent: obs.copy() for agent, obs in obs_dict.items()})
        act_batch.append({agent: act.cpu().numpy() for agent, act in actions.items()})
        
        # 执行动作（需要转换为numpy数组）
        actions_np = {agent: act.cpu().numpy() for agent, act in actions.items()}
        next_obs_dict, rewards, terminations, truncations, _ = env.step(actions_np)
        
        # 更新状态
        obs_dict = next_obs_dict
        done = any(terminations.values()) or any(truncations.values())
    
    return obs_batch, act_batch


class GAILDiscriminator:
    def __init__(self, obs_dim, act_dim, hidden_dim, device):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.n_agents = obs_dim // act_dim  # 假设 obs_dim 是所有智能体的观测维度拼接而成

        # 判别器网络
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_dim + act_dim, hidden_dim),  # 输入维度应该是 obs_dim + act_dim
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train_with_offline_data(self, expert_data_path):
        """
        使用离线数据训练 GAIL 判别器
        Args:
            expert_data_path (str): .npz 文件路径
        """
        # 加载离线数据
        data = np.load(expert_data_path)
        obs = torch.tensor(data["obs"], dtype=torch.float32).to(self.device)
        act = torch.tensor(data["act"], dtype=torch.float32).to(self.device)

        # 检查动作维度是否为所有智能体的拼接
        if obs.shape[1] == self.obs_dim * self.n_agents:
            print("检测到观测维度为所有智能体的拼接，拆分为单个智能体的观测")
            obs = obs.reshape(-1, self.n_agents, self.obs_dim)
            act = act.reshape(-1, self.n_agents, self.act_dim)

        # 构造输入
        inputs = torch.cat([obs[:, 0, :], act[:, 0, :]], dim=1)  # 仅使用第一个智能体的数据作为示例
        labels = torch.ones(inputs.shape[0], 1).to(self.device)

        # 训练判别器
        self.model.train()
        for _ in range(1000):
            self.optimizer.zero_grad()
            predictions = self.model(inputs)
            loss = torch.nn.functional.binary_cross_entropy(predictions, labels)
            loss.backward()
            self.optimizer.step()


