import sys
import os

# 添加项目根目录到 Python 模块搜索路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# updated_collect_expert_dataset.py
# collect_expert_dataset.py
import os
import numpy as np
from tqdm import trange
import torch

from pettingzoo.mpe import simple_adversary_v3
from agents.maddpg_agent import MADDPGAgent

# ===== 创建环境 =====
def create_env(render=False):
    env = simple_adversary_v3.parallel_env(
        continuous_actions=True,
        max_cycles=125,
        render_mode="rgb_array" if render else None
    )
    env.reset()
    env.max_cycles = 125
    return env

# ===== 加载训练好的策略 =====
def load_agents(model_dir, n_agents, obs_dim, goal_dim, act_dim, device="cpu"):
    agents = []
    for i in range(n_agents):
        agent = MADDPGAgent(
            agent_id=i,
            obs_dim=obs_dim,
            goal_dim=goal_dim,
            act_dim=act_dim,
            n_agents=n_agents,
            device=device
        )
        actor_path = os.path.join(model_dir, f"agent_{i}_actor.pth")
        agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
        agent.actor.eval()
        agents.append(agent)
    return agents

# ===== 使用 actor 生成动作 =====
def expert_policy(agent_obs, agent_id, agents, goal_dim):
    obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
    goal_tensor = torch.zeros(goal_dim, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        action = agents[agent_id].actor(obs_tensor, goal_tensor).squeeze(0).cpu().numpy()
    return action

# ===== 数据采集函数 =====
def save_expert_dataset(env, agents, n_episodes=2000, save_path="datasets/maddpg_expert.npz"):
    print(f"\n🚀 采集中: 使用训练好的 MADDPG 策略作为专家，共 {n_episodes} episodes")
    obs_list, act_list = [], []

    for _ in trange(n_episodes):
        obs_dict, _ = env.reset()
        trajectory_obs, trajectory_acts = [], []

        for _ in range(env.max_cycles):
            actions = {}
            for i, agent_id in enumerate(env.agents):
                act_dim = env.action_space(agent_id).shape[0]
                goal_dim = 2  # 目标维度，如无目标用零向量
                action = expert_policy(obs_dict[agent_id], i, agents, goal_dim)
                actions[agent_id] = action

            next_obs, rewards, terms, truncs, _ = env.step(actions)
            trajectory_obs.append(obs_dict)
            trajectory_acts.append(actions)
            obs_dict = next_obs

            if all(terms.values()) or all(truncs.values()):
                break

        try:
            obs_episode = []
            act_episode = []
            for step_obs, step_acts in zip(trajectory_obs, trajectory_acts):
                obs_step = [step_obs[a] for a in env.possible_agents]
                act_step = [step_acts[a] for a in env.possible_agents]
                obs_episode.append(np.concatenate(obs_step))
                act_episode.append(np.concatenate(act_step))
            obs_list.append(np.array(obs_episode))
            act_list.append(np.array(act_episode))
        except Exception as e:
            print(f"⚠️ 跳过无效轨迹: {e}")
            continue

    if not obs_list:
        raise RuntimeError("没有采集到有效的专家轨迹")

    obs_all = np.concatenate(obs_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, obs=obs_all, act=act_all)
    print(f"✅ 专家轨迹已保存至: {save_path}")
    print(f"📊 数据维度: obs {obs_all.shape}, act {act_all.shape}")

# ===== 主入口 =====
if __name__ == "__main__":
    env = create_env(render=False)
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    print(f"👀 真实 obs 维度: {obs_dim}")
    act_dim = env.action_space(env.agents[0]).shape[0]
    goal_dim = 2
    agents = load_agents("checkpoints/maddpg", len(env.agents), obs_dim, goal_dim, act_dim)
    save_expert_dataset(env, agents, n_episodes=200, save_path="datasets/maddpg_expert.npz")
    from visualize import render_expert_gif
    render_expert_gif(env, expert_policy, gif_path="results/expert_behavior.gif")