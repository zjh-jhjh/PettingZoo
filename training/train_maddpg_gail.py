# -*- coding: utf-8 -*
import sys
import os

# 添加项目根目录到 Python 模块搜索路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import torch
import argparse
from utils.replay_buffer import ReplayBuffer
from training.trainer import Trainer
from utils.env_utils import create_env
from agents.maddpg_agent import MADDPGAgent  # Ensure MADDPGAgent is imported
from training.gail_module import GAILDiscriminator
from collections import defaultdict  # Ensure defaultdict is imported

# 配置
N_GOOD = 2
#N_AGENTS = N_GOOD + 1


GOAL_DIM = 2  # MPE环境不需要目标维度
ACT_DIM = 5
LATENT_DIM = 64
HIDDEN_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="full", choices=["baseline", "gail", "encoder", "full"],
                        help="实验方法：baseline / gail / encoder / full")
    parser.add_argument("--episodes", type=int, default=10000, help="训练的总回合数")
    parser.add_argument("--logdir", type=str, default="logs/runs", help="TensorBoard 保存路径前缀")
    return parser.parse_args()
def main():
    args = parse_args()
    use_gail = args.method in ["gail", "full"]
    use_encoder = args.method in ["encoder", "full"]

    # === 1️⃣ 创建环境 ===
    env = create_env(use_encoder=use_encoder)
    env.reset()
    agent_ids = list(env.agents)  # ✅ 用这个固定顺序
    print("环境中的 agents:", agent_ids)

    n_agents = len(agent_ids)
    act_dim = env.action_space(agent_ids[0]).shape[0]
    latent_dim = 64
    hidden_dim = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs_dims = {}
    goal_dims = {}

    for agent_id in agent_ids:
        raw_obs_dim = env.observation_space(agent_id).shape[0]

        if use_encoder:
            # 环境返回的 obs 包含 goal，需要手动分离
            obs_dim = raw_obs_dim - 2  # 减去 goal 的维度
            goal_dim = 2 if "agent" in agent_id or "good" in agent_id or "survivor" in agent_id else 0
        else:
            # baseline 模式下 obs 已经拼接
            obs_dim = raw_obs_dim
            goal_dim = 0

        obs_dims[agent_id] = obs_dim
        goal_dims[agent_id] = goal_dim
        print(f"[Init Actor] Agent {agent_id} obs_dim={obs_dim}, goal_dim={goal_dim}")

    total_obs_dim = sum(obs_dims.values())
    all_obs_dims = [obs_dims[a] for a in agent_ids]   # ✅ 用固定 agent_ids
    all_goal_dims = [goal_dims[a] for a in agent_ids] # ✅

    agents = []
    for i, agent_id in enumerate(agent_ids):
        agents.append(
            MADDPGAgent(
                agent_id=i,
                obs_dim=obs_dims[agent_id],
                goal_dim=goal_dims[agent_id],
                act_dim=act_dim,
                n_agents=n_agents,
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                device=device,
                use_encoder=use_encoder,
                total_obs_dim=total_obs_dim,
                all_obs_dims=all_obs_dims,
                all_goal_dims=all_goal_dims,
            )
        )
        print(f"🧠 [Agent {i}] using obs_dim={obs_dims[agent_id]}, encoder={use_encoder}")
        print(f"🧠 [Agent {i}] actor type: {type(agents[-1].actor)}")
        # print(f"✅ Agent {agent_id}: obs_dim={obs_dims[agent_id]}, goal_dim={goal_dims[agent_id]}")


    # === 5️⃣ 初始化 GAIL 判别器 ===
    max_obs = max(obs_dims.values())
    max_goal = max(goal_dims.values())
    # 初始化 GAIL 判别器（统一维度：最大 obs + goal + act）
    if use_gail:
        gail_disc = GAILDiscriminator(
            obs_dim=max_obs,
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            device=device
        )
        env.unwrapped.scenario.gail_discriminator = gail_disc
        env.unwrapped.scenario.use_gail = True
    else:
        gail_disc = None

    # === 6️⃣ 初始化 ReplayBuffer ===
    buffer = ReplayBuffer(
        buffer_size=100_000,
        obs_dim=max_obs + max_goal,
        goal_dim=max_goal,
        act_dim=act_dim,
        n_agents=n_agents,
        device=device
    )

    # === 7️⃣ 初始化 Trainer ===
    writer_dir = os.path.join(args.logdir, f"maddpg_{args.method}")
    trainer = Trainer(
        env=env,
        agents=agents,
        buffer=buffer,
        gail_disc=gail_disc,
        batch_size=512,
        train_freq=100,
        gail_freq=100,
        max_steps=100_000,
        eval_freq=1000,
        max_cycles=25,
        device=device,
        obs_dim=max_obs,
        goal_dim=max_goal,
        tensorboard_logdir=writer_dir,
        method=args.method,
        max_episodes=args.episodes
    )

    trainer.run()


def run(self):
    """
    主训练循环
    """
    for step in range(self.max_steps):
        obs_dict = self.env.reset()
        print(f"obs_dict: {obs_dict}")  # 添加打印语句以检查结构
        episode_data = defaultdict(list)

        for _ in range(self.max_cycles):
            obs = [torch.tensor(obs_dict[agent], dtype=torch.float32) for agent in self.env.agents]
            goal = [torch.zeros(self.goal_dim, dtype=torch.float32) for _ in self.env.agents]  # 由于MPE环境没有目标，使用零向量


if __name__ == "__main__":
    main()
