import os
import numpy as np
from tqdm import trange
from pettingzoo.mpe import simple_adversary_v3  # 改回使用 mpe
from training.expert_policy import expert_policy
from training.gail_module import collect_trajectory_data


def create_env():
    # 使用默认参数创建环境，这样可以确保动作空间正确
    env = simple_adversary_v3.parallel_env(
        continuous_actions=True,
        render_mode=None
    )
    # 初始化环境
    env.reset()
    return env


def save_expert_dataset(env, policy_fn, n_episodes=100, save_path="expert_data.npz"):
    print(f"🚀 采集专家轨迹 {n_episodes} episodes ...")
    # 打印环境信息
    print("环境信息:")
    print(f"智能体列表: {env.possible_agents}")
    for agent in env.possible_agents:
        print(f"Agent {agent}:")
        print(f"- 观测空间: {env.observation_space(agent)}")
        print(f"- 动作空间: {env.action_space(agent)}")
    
    obs_list = []
    act_list = []

    for _ in trange(n_episodes):
        trajectory_obs, trajectory_acts = collect_trajectory_data(env, policy_fn)
        if not trajectory_obs:  # 跳过空轨迹
            continue
            
        # 处理每一步的观测和动作
        try:
            episode_obs = []
            episode_acts = []
            
            # 遍历轨迹中的每一步
            for step_obs, step_acts in zip(trajectory_obs, trajectory_acts):
                # 将所有智能体的观测和动作连接起来
                step_obs_list = []
                step_acts_list = []
                
                # 按固定顺序处理智能体
                for agent in env.possible_agents:
                    step_obs_list.append(step_obs[agent])
                    step_acts_list.append(step_acts[agent])
                
                # 连接所有智能体的数据
                step_obs_concat = np.concatenate(step_obs_list)
                step_acts_concat = np.concatenate(step_acts_list)
                
                episode_obs.append(step_obs_concat)
                episode_acts.append(step_acts_concat)
            
            # 将整个轨迹转换为数组
            episode_obs = np.array(episode_obs)
            episode_acts = np.array(episode_acts)
            
            obs_list.append(episode_obs)
            act_list.append(episode_acts)
            
        except Exception as e:
            print(f"跳过无效轨迹: {e}")
            continue

    if not obs_list:
        raise RuntimeError("没有收集到有效的轨迹数据")

    # 连接所有轨迹数据
    obs_all = np.concatenate(obs_list, axis=0)
    act_all = np.concatenate(act_list, axis=0)

    # 保存数据
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, obs=obs_all, act=act_all)
    print(f"✅ 已保存专家数据到: {save_path}")
    print(f"📊 观测维度: {obs_all.shape}, 动作维度: {act_all.shape}")


if __name__ == "__main__":
    env = create_env()
    save_path = "datasets/expert_simple_adversary.npz"
    save_expert_dataset(env, expert_policy, n_episodes=200, save_path=save_path)
