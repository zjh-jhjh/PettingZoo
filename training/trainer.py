import torch
import copy
import numpy as np
import imageio.v2 as imageio
from utils.env_utils import create_env
from training.expert_policy import expert_policy
from training.gail_module import collect_trajectory_data 


class Trainer:
    def __init__(self, env, agents, buffer, gail_disc, batch_size, train_freq, gail_freq, max_steps, eval_freq, max_cycles, device, obs_dim, goal_dim, tensorboard_logdir=None,max_episodes=2000):
        self.env = env
        self.agents = agents
        self.buffer = buffer
        self.gail_disc = gail_disc
        self.batch_size = batch_size
        self.train_freq = train_freq
        self.gail_freq = gail_freq
        self.max_steps = max_steps
        self.eval_freq = eval_freq
        self.max_cycles = max_cycles
        self.n_agents = len(agents)
        self.device = device
        self.obs_dim = obs_dim  # 添加 obs_dim 属性
        self.goal_dim = goal_dim  # 添加 goal_dim 属性
        self.max_episodes = max_episodes
        self.gail = None  # Initialize gail attribute
        self.actor = None  # Initialize actor attribute
        self.agent_name_to_index = {agent_id: i for i, agent_id in enumerate(env.possible_agents)}

        # 初始化 TensorBoard 日志记录器（如果提供了路径）
        if tensorboard_logdir:
            from torch.utils.tensorboard.writer import SummaryWriter
            self.writer = SummaryWriter(log_dir=tensorboard_logdir)
        else:
            self.writer = None

    def load_offline_data(self, expert_data_path):
        """加载离线数据到 ReplayBuffer"""
        data = np.load(expert_data_path)
        obs = data["obs"]
        act = data["act"]

        # 修复观测维度
        expected_obs_dim = self.buffer.n_agents * self.buffer.obs_dim
        if obs.shape[1] != expected_obs_dim:
            print(f"修复观测维度: {obs.shape[1]} -> {expected_obs_dim}")
            if obs.shape[1] > expected_obs_dim:
                obs = obs[:, :expected_obs_dim]  # 截断多余的维度
            else:
                padding = np.zeros((obs.shape[0], expected_obs_dim - obs.shape[1]))
                obs = np.concatenate([obs, padding], axis=1)  # 填充缺失的维度

        # 计算每个智能体的观测维度
        obs_dim_per_agent = obs.shape[1] // self.buffer.n_agents

        # 将观测拆分为每个智能体的观测
        obs_per_agent = obs.reshape(obs.shape[0], self.buffer.n_agents, obs_dim_per_agent)

        # 将数据存入 ReplayBuffer
        for i in range(obs.shape[0]):
            for agent_idx in range(self.buffer.n_agents):
                self.buffer.add(
                    obs_per_agent[i, agent_idx],
                    act[i, agent_idx],
                    np.zeros(obs_dim_per_agent),
                    np.zeros(self.buffer.goal_dim),
                    None, None, None
                )

    def run(self):
        step_count = 0
        for episode in range(self.max_episodes):
            obs_dict, _ = self.env.reset()
            episode_reward = np.zeros(self.n_agents)

            for t in range(self.max_cycles):
                obs = [torch.tensor(obs_dict[agent][:self.obs_dim], dtype=torch.float32) for agent in self.env.agents]
                goal = [torch.tensor(obs_dict[agent][-self.goal_dim:], dtype=torch.float32) for agent in self.env.agents]

                # 策略选择动作（带探索）
                action_dict = {}
                for i, agent in enumerate(self.agents):
                    if agent.goal_dim > 0:
                        goal_input = goal[i]
                    else:
                        goal_input = None

                    action = agent.select_action(obs[i], goal_input, explore=True)
                    action_dict[self.env.agents[i]] = action.detach().cpu().numpy()

                # 与环境交互
                next_obs_dict, rewards, terminations, truncations, _ = self.env.step(action_dict)
                dones = [float(terminations[a]) for a in self.env.agents]
                rew = [rewards.get(agent, 0.0) for agent in self.env.agents]
                if len(rew) == 0:
                    print("Warning: rewards empty!")
                else:
                    episode_reward += np.array(rew)
                # print(f"Episode {episode} rewards dict: {rewards}")
                # print(f"Agents: {self.env.agents}")
                # print(f"rew: {rew}")

                if len(self.env.agents) > 0:
                # 存储 transition
                    max_obs_dim = max([obs_dict[a].shape[0] for a in self.env.agents])  # 找到最大的观测维度
                    episode_reward += np.array(rew)
                else:
                    print("Warning: No agents found in the environment!")
                    break

                # 构造 obs+goal 并 padding（每个智能体一份）
                obs_with_goal_np = []
                next_obs_with_goal_np = []
                actions_np = []

                for i, agent_id in enumerate(self.env.agents):
                    obs_vec = obs_dict[agent_id]
                    next_obs_vec = next_obs_dict[agent_id]

                    # 分割 obs 与 goal
                    obs_part = obs_vec[:self.obs_dim]
                    goal_part = obs_vec[-self.goal_dim:] if self.goal_dim > 0 else np.array([])

                    next_obs_part = next_obs_vec[:self.obs_dim]
                    next_goal_part = next_obs_vec[-self.goal_dim:] if self.goal_dim > 0 else np.array([])

                    # 拼接 obs + goal，padding 到统一长度
                    obs_g = np.concatenate([obs_part, goal_part])
                    next_obs_g = np.concatenate([next_obs_part, next_goal_part])

                    # padding 到 buffer 设定的 obs_dim（通常是 max(obs+goal)）
                    obs_g = self.pad_to_fixed_length(obs_g, self.buffer.obs_dim)
                    next_obs_g = self.pad_to_fixed_length(next_obs_g, self.buffer.obs_dim)

                    obs_with_goal_np.append(obs_g)
                    next_obs_with_goal_np.append(next_obs_g)
                    actions_np.append(action_dict[agent_id])

                # 存入 buffer
                for i in range(self.n_agents):
                    self.buffer.add(
                        obs=obs_with_goal_np[i],
                        act=actions_np[i],
                        next_obs=next_obs_with_goal_np[i],
                        next_goals=np.zeros(self.goal_dim, dtype=np.float32),  # 若已含入 obs 可为占位
                        reward=rewards.get(self.env.agents[i], 0.0),
                        done=float(terminations.get(self.env.agents[i], False)),
                        info=None
                    )
                # 更新状态
                obs_dict = next_obs_dict
                step_count += 1

                if all(terminations.values()) or all(truncations.values()):
                    break

            # ✅ 策略训练
            if step_count > self.batch_size and episode % self.train_freq == 0:
                batch = self.buffer.sample(self.batch_size)
                for agent in self.agents:
                    actor_loss, critic_loss = agent.update(batch, self.agents)
                    agent.soft_update()

                    if hasattr(self, "writer"):
                        if self.writer is not None:
                            self.writer.add_scalar(f"loss/actor_{agent.agent_id}", actor_loss, step_count)
                            self.writer.add_scalar(f"loss/critic_{agent.agent_id}", critic_loss, step_count)

            # ✅ GAIL 判别器训练
            if self.gail and step_count > self.batch_size and episode % self.gail_freq == 0:
                obs_exp, act_exp = self.sample_expert_data()
                obs_pol, act_pol = self.sample_agent_data()
                gail_loss = self.gail.train_step((obs_exp, act_exp), (obs_pol, act_pol))
                if hasattr(self, "writer"):
                    if self.writer is not None:
                        self.writer.add_scalar("loss/gail", gail_loss, step_count)

            # ✅ 策略评估与可视化
            if episode % self.eval_freq == 0:
                self.env_step_count = step_count
                self.evaluate(save_gif=True, gif_path=f"results/episode_{episode}.gif")

                if hasattr(self, "writer"):
                    mean_episode_reward = episode_reward.mean()
                    if self.writer is not None:
                        self.writer.add_scalar("eval/total_episode_reward", mean_episode_reward, step_count)

            print(f"✅ Episode {episode} done — steps so far: {step_count}")

            # 每隔 eval_freq 步保存 agent 模型
            if step_count % self.eval_freq == 0:
                for i, agent in enumerate(self.agents):
                    save_path = f"logs/checkpoints/maddpg/agent_{i}_actor.pth"
                    torch.save(agent.actor.state_dict(), save_path)
                print(f"💾 已保存 agent 策略至 checkpoints/maddpg/")



    def sample_agent_data(self):
        """从 buffer 中采样用于 GAIL 判别器训练"""
        batch = self.buffer.sample(self.batch_size)
        obs = batch["obs"]
        goals = batch["goals"]
        acts = batch["actions"]
        # 合并所有 agent 的样本为 batch
        obs_all = torch.cat([torch.cat([obs[i], goals[i]], dim=-1) for i in range(self.n_agents)], dim=0)
        act_all = torch.cat([acts[i] for i in range(self.n_agents)], dim=0)
        return obs_all, act_all

    #对 obs + goal 向量进行右侧 padding
    def pad_to_fixed_length(self, x: np.ndarray, target_dim: int) -> np.ndarray:
        if x.shape[0] == target_dim:
            return x
        elif x.shape[0] < target_dim:
            padding = np.zeros(target_dim - x.shape[0], dtype=np.float32)
            return np.concatenate([x, padding])
        else:
            raise ValueError(f"Input too long: {x.shape[0]} > {target_dim}")


    def sample_expert_data(self):
        """从专家策略采集一批真实 (obs, act) 轨迹"""
        obs_batch, act_batch = collect_trajectory_data(self.env, expert_policy)

        # 拼接所有 agent 的数据作为 GAIL 训练输入
        obs_all = torch.cat([
            torch.tensor(obs_batch[i], dtype=torch.float32) for i in range(self.n_agents)
        ], dim=0).to(self.device)

        act_all = torch.cat([
            torch.tensor(act_batch[i], dtype=torch.float32) for i in range(self.n_agents)
        ], dim=0).to(self.device)

        return obs_all, act_all


    def evaluate(self, n_episodes=5, verbose=True, save_gif=False, gif_path="eval.gif"):
        total_rewards = []
        goal_hits = []
        env_render = copy.deepcopy(self.env)
        obs_dict, _ = env_render.reset()
        obs_dict, _ = env_render.reset()

        obs = {
            agent_id: torch.tensor(obs_dict[agent_id][:self.obs_dim], dtype=torch.float32).to(self.device)
            for agent_id in env_render.agents
        }

        goal = {
            agent_id: torch.tensor(obs_dict[agent_id][-self.goal_dim:], dtype=torch.float32).to(self.device)
            if self.goal_dim > 0 else None
            for agent_id in env_render.agents
        }

        for ep in range(n_episodes):
            env_render = self.env
            if save_gif:
                env_render = create_env(render=True)  # 创建渲染版环境
                obs_dict, _ = env_render.reset()
            else:
                obs_dict, _ = self.env.reset()

            episode_reward = np.zeros(self.n_agents)
            episode_goal_hit = np.zeros(self.n_agents)
            frames = []

            for _ in range(self.max_cycles):
                if save_gif:
                    frame = env_render.render()
                    frames.append(frame)


                action_dict = {}
                for agent_id in env_render.agents:
                    i = self.agent_name_to_index[agent_id]
                    obs_tensor = obs[agent_id]
                    goal_tensor = goal[agent_id]

                    action = self.agents[i].select_action(obs_tensor, goal_tensor, explore=False)
                    action_dict[agent_id] = action.detach().cpu().numpy()

                obs_next, rewards, dones, truncs, _ = env_render.step(action_dict)
                if len(self.env.agents) > 0:
                    episode_reward += np.array([rewards[a] for a in env_render.agents])
                else:
                    print("Warning: No agents found in the environment!")
                    break

                for i, a in enumerate(env_render.agents):
                    if np.linalg.norm(obs_dict[a][-self.goal_dim:]) < 0.1:
                        episode_goal_hit[i] = 1

                obs_dict = obs_next
                if all(dones.values()) or all(truncs.values()):
                    break

            total_rewards.append(episode_reward)
            goal_hits.append(episode_goal_hit)

            if save_gif and ep == 0:
                imageio.mimsave(gif_path, frames, duration=0.1)
                print(f"🎞️  Saved trajectory gif to {gif_path}")

        avg_reward = np.mean(total_rewards, axis=0)
        avg_hit = np.mean(goal_hits, axis=0)
        if verbose:
            print(f"\n🧪 Evaluation:")
            for i in range(self.n_agents):
                print(f"  Agent {i}: Reward = {avg_reward[i]:.2f}, Goal hit rate = {avg_hit[i]*100:.1f}%")
        return avg_reward, avg_hit


