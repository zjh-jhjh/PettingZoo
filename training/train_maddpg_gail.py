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
N_AGENTS = N_GOOD + 1


GOAL_DIM = 2  # MPE环境不需要目标维度
ACT_DIM = 5
LATENT_DIM = 64
HIDDEN_DIM = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="full", choices=["baseline", "gail", "encoder", "full"],
                        help="实验方法：baseline / gail / encoder / full")
    parser.add_argument("--logdir", type=str, default="logs/runs", help="TensorBoard 保存路径前缀")
    return parser.parse_args()
def main():
    args = parse_args()
    use_gail = args.method in ["gail", "full"]
    use_encoder = args.method in ["encoder", "full"]

    env = create_env()
    env.reset()
    n_agents = len(env.agents)
    act_dim = env.action_space(env.agents[0]).shape[0]  # 动作维度从环境获取
    latent_dim = 64
    hidden_dim = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(env.step.__module__)
    print(env.step.__qualname__)

    # 分别记录每个 agent 的 obs_dim 和 goal_dim
    obs_dims = {}
    goal_dims = {}
    for agent_id in env.agents:
        obs_dim = env.observation_space(agent_id).shape[0]
        obs_dims[agent_id] = obs_dim 
        goal_dims[agent_id] = 2 if "agent" in agent_id else 0  # good agent 有目标，adversary 没有
        print(f"[DEBUG] Agent {agent_id} obs_dim: {obs_dims[agent_id]}, goal_dim: {goal_dims[agent_id]}, act_dim: {act_dim}")
    # 假设你已经有了 obs_dims 字典
    total_obs_dim = sum([obs_dims[agent_id] for agent_id in env.agents])
    all_obs_dims = [obs_dims[agent_id] for agent_id in env.agents]
    all_goal_dims = [goal_dims[agent_id] for agent_id in env.agents]
    # 初始化智能体
    agents = []
    for i, agent_id in enumerate(env.agents):
        agent_obs_dim = obs_dims[agent_id]  # ✅
        agent_goal_dim = goal_dims[agent_id]  # ✅
        print(f"[Init Actor] Agent {agent_id} obs_dim={agent_obs_dim}, goal_dim={agent_goal_dim}")
        agents.append(
            MADDPGAgent(
                agent_id=i,
                obs_dim=agent_obs_dim,
                goal_dim=agent_goal_dim,
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
        # print(f"✅ Agent {agent_id}: obs_dim={obs_dims[agent_id]}, goal_dim={goal_dims[agent_id]}")

    for i, agent_id in enumerate(env.agents):
        print(f"Agent {agent_id} obs dim: {env.observation_space(agent_id).shape[0]}")

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
    else:
        gail_disc = None

    # 初始化经验回放池（也用最大 obs + goal 维度）
    buffer = ReplayBuffer(
        buffer_size=100_000,
        obs_dim=max_obs + max_goal,
        goal_dim=max_goal,
        act_dim=act_dim,
        n_agents=n_agents,
        device=device
    )

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
        tensorboard_logdir=writer_dir
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
