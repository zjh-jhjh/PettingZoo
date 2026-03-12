import torch
import copy
import os
import sys
import numpy as np
import imageio.v2 as imageio

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env_utils import create_env
from training.expert_policy import expert_policy
from training.gail_module import collect_trajectory_data
from modules.actor import GoalConditionedActor


class Trainer:
    def __init__(self, env, agents, buffer, gail_disc, batch_size, train_freq, gail_freq, max_steps, eval_freq, max_cycles, device, obs_dim, goal_dim,method="baseline", tensorboard_logdir=None,max_episodes=2000):
        self.env = env
        self.method = method
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
        self.use_encoder_obs = getattr(self.env.unwrapped, "use_encoder_obs", False)
        step_count = 0
        for episode in range(self.max_episodes):
            obs_dict, _ = self.env.reset()
            episode_reward = np.zeros(self.n_agents)

            for t in range(self.max_cycles):
                obs = []
                goal = []
                for i, agent_id in enumerate(self.env.agents):
                    agent_obs_dim = self.agents[i].obs_dim
                    agent_goal_dim = self.agents[i].goal_dim
                    obs_vec = obs_dict[agent_id]
                    if agent_id.startswith('adversary'):
                        # 对于 adversary，使用全部 8 维观察
                        obs_i = torch.tensor(obs_vec, dtype=torch.float32)
                        goal_i = torch.tensor([])  # 空目标
                    else:
                        # 对于 agent，使用 8 维观察拼接 2 维目标，总共 10 维
                        if self.method in ["baseline", "gail"]:
                            # ✅ baseline 模式：goal 已经拼进 obs，直接取前 obs_dim 部分即可
                            obs_i = torch.tensor(obs_vec[:self.obs_dim], dtype=torch.float32)
                            goal_i = torch.tensor([], dtype=torch.float32)  # 空 goal（不输入 encoder）

                        elif self.method in ["encoder", "full"]:
                            # ✅ encoder 模式：goal 不拼进 obs，从 agent.goal 取（即 env 内部目标位置）
                            agent_goal_pos = torch.tensor(self.env.unwrapped.world.agents[i].goal.state.p_pos, dtype=torch.float32)
                            obs_i = torch.tensor(obs_vec, dtype=torch.float32)  # 仅状态部分
                            goal_i = agent_goal_pos

                        else:
                            raise ValueError(f"Unknown training method: {self.method}")

                    obs.append(obs_i)
                    goal.append(goal_i)

                # 策略选择动作（带探索）
                # === 策略选择动作（带探索）===
                action_dict = {}
                for i, agent in enumerate(self.agents):
                    agent_name = self.env.agents[i]
                    raw_obs = obs_dict[agent_name]

                    # === 关键逻辑：根据模式决定如何拆分 ===
                    if agent.use_encoder:
                        # ✅ encoder 模式：obs 是完整环境观测（12维），goal 单独取最后 2维
                        agent_obs = raw_obs[:agent.obs_dim]  # 取前面的观察部分
                        goal_input = self.env.unwrapped.world.agents[i].goal.state.p_pos - self.env.unwrapped.world.agents[i].state.p_pos  # 由环境获取
                    else:
                        # ✅ baseline 模式：goal 已拼入 obs（10维），直接作为输入
                        agent_obs = raw_obs
                        goal_input = None

                    # === 转 tensor ===
                    agent_obs = torch.tensor(agent_obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    goal_input = (
                        torch.tensor(goal_input, dtype=torch.float32, device=self.device).unsqueeze(0)
                        if goal_input is not None else None
                    )

                    # === 策略选择动作 ===
                    action = agent.select_action(agent_obs, goal_input, explore=True)
                    # 确保动作形状正确 (5,) 而不是 (1, 5)
                    action_dict[agent_name] = action.detach().cpu().numpy().flatten()




    # print("🤖 当前 agents:", self.env.agents)
                # print("输出动作", action_dict)
                # 与环境交互
                next_obs_dict, rewards, terminations, truncations, _ = self.env.step(action_dict)
                # print("🎯 输出 reward:", rewards)
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
                    agent_obs_dim = self.agents[i].obs_dim
                    agent_goal_dim = self.agents[i].goal_dim

                    obs_part = obs_vec[:agent_obs_dim]
                    goal_part = obs_vec[-agent_goal_dim:] if agent_goal_dim > 0 else np.array([])
                    # print(
                    #     f"[CHECK] Agent {agent_id} full obs_vec.shape: {len(obs_vec)}, expect obs_dim={agent_obs_dim}, goal_dim={agent_goal_dim}")

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

            # ✅ 记录每个episode的奖励（用于绘制训练收敛曲线）
            if hasattr(self, "writer") and self.writer is not None:
                mean_episode_reward = episode_reward.mean()
                # 使用episode数作为x轴，而不是step_count
                self.writer.add_scalar("train/episode_reward", mean_episode_reward, episode)
                # 记录每个智能体的奖励
                for i in range(self.n_agents):
                    self.writer.add_scalar(f"train/agent_{i}_reward", episode_reward[i], episode)

            # ✅ 策略评估与可视化
            if episode % self.eval_freq == 0:
                self.env_step_count = step_count
                avg_reward, avg_hit = self.evaluate(save_gif=True, gif_path=f"results/episode_{episode}.gif")

                if hasattr(self, "writer") and self.writer is not None:
                    mean_eval_reward = avg_reward.mean()
                    self.writer.add_scalar("eval/total_episode_reward", mean_eval_reward, episode)
                    # 记录每个智能体的评估奖励
                    for i in range(self.n_agents):
                        self.writer.add_scalar(f"eval/agent_{i}_reward", avg_reward[i], episode)
                        self.writer.add_scalar(f"eval/agent_{i}_goal_hit_rate", avg_hit[i], episode)

            # print(f"✅ Episode {episode} done — steps so far: {step_count}")

            # 每隔 eval_freq 步保存 agent 模型
            if step_count % self.eval_freq == 0:
                save_dir = os.path.join("logs/checkpoints", self.method)
                os.makedirs(save_dir, exist_ok=True)
                for i, agent in enumerate(self.agents):
                    save_path = os.path.join(save_dir, f"agent_{i}_actor.pth")
                    torch.save(agent.actor.state_dict(), save_path)
                    print(f"💾 已保存 agent 策略至 {save_path}")

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
    def pad_to_fixed_length(self, x: np.ndarray, target_dim: int, pad_value: float = 0.0) -> np.ndarray:
        """
        将输入向量 x 补齐（或截断）到固定长度 target_dim。
        - baseline/gail 模式：x 可能已经是 obs+goal 拼接后的完整输入；
        - encoder/full 模式：x 可能仅包含 obs，因此需要 padding。
        """
        # 确保输入是一维向量
        if x.ndim > 1:
            x = x.flatten()

        x_len = x.shape[0]

        if x_len == target_dim:
            # ✅ 已经匹配，直接返回
            return x.astype(np.float32)

        elif x_len < target_dim:
            # ✅ 长度不足，右侧补零（或 pad_value）
            padding = np.full(target_dim - x_len, pad_value, dtype=np.float32)
            x_padded = np.concatenate([x, padding])
            # print(f"[Pad] 输入长度 {x_len} < {target_dim}，右侧补 {target_dim - x_len} 维。")
            return x_padded

        else:
            # ⚠️ 长度超出时截断（防止 shape mismatch）
            print(f"[Warning] 输入长度 {x_len} > 目标维度 {target_dim}，已截断。")
            return x[:target_dim].astype(np.float32)



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

                    # 每个 agent 单独处理 goal_dim
                    goal_dim = self.agents[i].goal_dim if hasattr(self.agents[i], "goal_dim") else 0
                    if self.method in ["encoder", "full"]:
                        goal_tensor = torch.tensor(env_render.unwrapped.world.agents[i].goal.state.p_pos, dtype=torch.float32)
                    else:
                        goal_tensor = torch.tensor(obs_dict[agent_id][-self.goal_dim:], dtype=torch.float32)

                    action = self.agents[i].select_action(obs_tensor, goal_tensor, explore=False)
                    action_dict[agent_id] = action.detach().cpu().numpy()

                obs_next, rewards, dones, truncs, _ = env_render.step(action_dict)
                if len(self.env.agents) > 0:
                    episode_reward += np.array([rewards[a] for a in env_render.agents])
                else:
                    print("Warning: No agents found in the environment!")
                    break

                for i, a in enumerate(env_render.agents):
                    if self.method == "baseline":
                        # baseline 模式下，goal 已经拼进 obs，直接判断是否接近目标
                        goal_rel = obs_dict[a][:2]
                    else:
                        # encoder 模式：goal 单独存储在 obs 尾部
                        goal_rel = obs_dict[a][-self.goal_dim:]
                    if np.linalg.norm(goal_rel) < 0.1:
                        episode_goal_hit[i] = 1.0

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


