import copy
import torch
import torch.nn as nn
import torch.optim as optim
from modules.actor import GoalConditionedActor
from modules.critic import CentralizedCritic
from modules.raw_actor import RawActor
from modules.raw_critic import RawCentralizedCritic


class MADDPGAgent:
    def __init__(self, agent_id, obs_dim, goal_dim, act_dim, n_agents,total_obs_dim,all_obs_dims,all_goal_dims,
                 latent_dim=64, hidden_dim=128, lr_actor=1e-3, lr_critic=1e-3,
                 tau=0.01, gamma=0.95, device="cpu", use_encoder=False):
        print(f"🧠 [Agent {agent_id}] using obs_dim={obs_dim}, encoder={use_encoder}")
        self.obs_dim = obs_dim  # 显式记录
        self.goal_dim = goal_dim
        self.agent_id = agent_id
        self.device = device
        self.n_agents = n_agents
        self.tau = tau
        self.gamma = gamma
        self.use_encoder = use_encoder
        self.total_obs_dim = total_obs_dim
        self.all_obs_dims = all_obs_dims
        self.all_goal_dims = all_goal_dims

        if use_encoder and goal_dim > 0:
            self.actor = GoalConditionedActor(obs_dim, goal_dim, act_dim, latent_dim, hidden_dim).to(device)
            self.critic = CentralizedCritic(all_obs_dims, all_goal_dims, act_dim, latent_dim, hidden_dim).to(device)
        else:
            self.actor = RawActor(obs_dim, act_dim, hidden_dim).to(device)
            self.critic = RawCentralizedCritic(obs_dims=all_obs_dims, act_dim=act_dim).to(device)

        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        print(f"🧠 [Agent {agent_id}] actor type: {type(self.actor)}")

    def select_action(self, obs, goal=None, explore=False, noise_std=0.1):
        self.actor.eval()
        obs = obs.unsqueeze(0).to(self.device)

        # actor_type = type(self.actor).__name__
        # print(
        #    f"[select_action] Agent {self.agent_id} using actor: {self.actor.__class__.__name__}, obs shape: {obs.shape}, goal: {goal.shape if goal is not None else None}")


        if isinstance(self.actor, GoalConditionedActor):
            assert goal is not None, f"❌ Agent {self.agent_id} 需要 goal，但收到 None"
            goal = goal.unsqueeze(0).to(self.device)
            action = self.actor(obs, goal).squeeze(0)
        else:
            # 防止错误：RawActor 不应接收 goal
            action = self.actor(obs).squeeze(0)

        if explore:
            noise = torch.normal(0, noise_std, size=action.shape).to(self.device)
            action = action + noise

        return action.clamp(-1, 1)

    def update(self, samples, agent_list):
        """
        更新当前 agent 的 actor 和 critic 网络
        samples: ReplayBuffer.sample() 返回的字典
        agent_list: 所有 agent 实例（用于 centralized critic）
        """
        obs, obs_next = samples["obs"], samples["obs_next"]
        goals, goals_next = samples["goals"], samples["goals_next"]
        actions, rewards, dones = samples["actions"], samples["rewards"], samples["dones"]

        obs_i = []
        obs_next_i = []
        goals_i = []
        goals_next_i = []

        for i in range(self.n_agents):
            agent_obs_dim = agent_list[i].obs_dim
            agent_goal_dim = agent_list[i].goal_dim

            obs_i.append(obs[i][:agent_obs_dim].to(self.device))
            obs_next_i.append(obs_next[i][:agent_obs_dim].to(self.device))
            goals_i.append(goals[i][:agent_goal_dim].to(self.device))
            goals_next_i.append(goals_next[i][:agent_goal_dim].to(self.device))

        actions_i = [actions[i].to(self.device) for i in range(self.n_agents)]
        rewards_i = rewards[self.agent_id].to(self.device).unsqueeze(1)
        dones_i = dones[self.agent_id].to(self.device).unsqueeze(1)

        # === 计算目标 Q 值 ===
        with torch.no_grad():
            next_actions = []
            for i, agent in enumerate(agent_list):
                # ✅ 确保输入 obs/goal 都是二维张量 [B, dim]
                obs_in = obs_next_i[i]
                goal_in = goals_next_i[i]
                if obs_in.dim() == 1:
                    obs_in = obs_in.unsqueeze(0)
                if goal_in.dim() == 1:
                    goal_in = goal_in.unsqueeze(0)

                # ✅ 调用 actor_target
                if agent.use_encoder:
                    next_action = agent.actor_target(obs_in, goal_in)
                else:
                    next_action = agent.actor_target(obs_in)

                # ✅ 保证输出动作也是 [B, act_dim]
                if next_action.dim() == 1:
                    next_action = next_action.unsqueeze(0)

                next_actions.append(next_action)



        # if self.use_encoder:
            if isinstance(self.critic_target, CentralizedCritic):
                q_next = self.critic_target(obs_next_i, goals_next_i, next_actions)
            else:
                q_next = self.critic_target(obs_next_i, next_actions)

            q_target = rewards_i + self.gamma * (1 - dones_i) * q_next

        # === 当前 Q 值 ===
        # if self.use_encoder:
        if isinstance(self.critic_target, CentralizedCritic):
            q_current = self.critic(obs_i, goals_i, actions_i)
        else:
            q_current = self.critic(obs_i, actions_i)

        # === Critic 更新 ===
        critic_loss = nn.MSELoss()(q_current, q_target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # === Actor 更新（仅当前 agent） ===
        self.actor.train()
        actions_pred = []
        for i in range(self.n_agents):
            if i == self.agent_id:
                obs_in = obs_i[i]
                goal_in = goals_i[i]
                if obs_in.dim() == 1:
                    obs_in = obs_in.unsqueeze(0)
                if goal_in.dim() == 1:
                    goal_in = goal_in.unsqueeze(0)

                if self.use_encoder:
                    action = self.actor(obs_in, goal_in)
                else:
                    action = self.actor(obs_in)
                actions_pred.append(action)
            else:
                actions_pred.append(actions_i[i].detach())



        # if self.use_encoder:
        if isinstance(self.critic_target, CentralizedCritic):
            actor_loss = -self.critic(obs_i, goals_i, actions_pred).mean()
        else:
            actor_loss = -self.critic(obs_i, actions_pred).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def soft_update(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def _extract_obs(self, agent, x):
        """
        从 obs+goal 拼接中提取 obs 部分。
        如果 agent 使用 encoder，则直接返回；
        如果使用 RawActor，则裁剪出 obs_dim。
        """
        x = x.to(self.device)
        if agent.use_encoder:
            return x
        else:
            return x[:agent.obs_dim] if x.dim() == 1 else x[:, :agent.obs_dim]

