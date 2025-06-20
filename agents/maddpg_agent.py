import copy
import torch
import torch.nn as nn
import torch.optim as optim
from modules.actor import GoalConditionedActor
from modules.critic import CentralizedCritic
from modules.raw_actor import RawActor
from modules.raw_critic import RawCentralizedCritic


class MADDPGAgent:
    def __init__(self, agent_id, obs_dim, goal_dim, act_dim, n_agents,
                 latent_dim=64, hidden_dim=128, lr_actor=1e-3, lr_critic=1e-3, tau=0.01, gamma=0.95, device="cpu",use_encoder=True):

        self.agent_id = agent_id
        self.device = device
        self.n_agents = n_agents
        self.tau = tau
        self.gamma = gamma

        # Actor network
        self.actor = GoalConditionedActor(obs_dim, goal_dim, act_dim, latent_dim, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Centralized critic (shared encoder inside)
        self.critic = CentralizedCritic(obs_dim, goal_dim, act_dim, n_agents, latent_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.use_encoder = use_encoder

        if use_encoder:
            self.actor = GoalConditionedActor(obs_dim, goal_dim, act_dim, latent_dim, hidden_dim).to(device)
            self.critic = CentralizedCritic(obs_dim, goal_dim, act_dim, n_agents, latent_dim, hidden_dim).to(device)
        else:
            self.actor = RawActor(obs_dim, act_dim, hidden_dim).to(device)
            self.critic = RawCentralizedCritic(obs_dim, act_dim, n_agents, hidden_dim).to(device)

        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

    def select_action(self, obs, goal, explore=False, noise_std=0.1):
        """
        obs: Tensor shape (obs_dim,)
        goal: Tensor shape (goal_dim,)
        """
        self.actor.eval()
        obs = obs.unsqueeze(0).to(self.device)
        goal = goal.unsqueeze(0).to(self.device)
        action = self.actor(obs, goal).squeeze(0)

        if explore:
            noise = torch.normal(0, noise_std, size=action.shape).to(self.device)
            action = action + noise
        return action.clamp(-1, 1)

    def update(self, samples, agent_list):
        """
        samples: dict with keys: obs, obs_next, goals, goals_next, actions, rewards, dones
            all shape (n_agents, batch, dim)
        agent_list: List[MADDPGAgent], 所有 agent 实例
        """
        obs, obs_next = samples["obs"], samples["obs_next"]
        goals, goals_next = samples["goals"], samples["goals_next"]
        actions, rewards, dones = samples["actions"], samples["rewards"], samples["dones"]

        # 转换为 list[Tensor] 每个元素 (B, dim)
        obs_i = [obs[i].to(self.device) for i in range(self.n_agents)]
        obs_next_i = [obs_next[i].to(self.device) for i in range(self.n_agents)]
        goals_i = [goals[i].to(self.device) for i in range(self.n_agents)]
        goals_next_i = [goals_next[i].to(self.device) for i in range(self.n_agents)]
        actions_i = [actions[i].to(self.device) for i in range(self.n_agents)]
        rewards_i = rewards[self.agent_id].to(self.device).unsqueeze(1)
        dones_i = dones[self.agent_id].to(self.device).unsqueeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            next_actions = [
                agent.actor_target(obs_next_i[i], goals_next_i[i])
                for i, agent in enumerate(agent_list)
            ]
            q_next = self.critic_target(obs_next_i, goals_next_i, next_actions)
            q_target = rewards_i + self.gamma * (1 - dones_i) * q_next

        # 当前 Q 值
        q_current = self.critic(obs_i, goals_i, actions_i)

        # critic loss
        critic_loss = nn.MSELoss()(q_current, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新 actor：只更新自己的 actor
        self.actor.train()
        actions_pred = [
            self.actor(obs_i[i], goals_i[i]) if i == self.agent_id else actions_i[i].detach()
            for i in range(self.n_agents)
        ]
        actor_loss = -self.critic(obs_i, goals_i, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def soft_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
