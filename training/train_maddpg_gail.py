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
OBS_DIM = 8  # 修改为实际的观测维度
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
    agent_ids = env.agents
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    print(f"👀 真实 obs 维度: {obs_dim}")
    # Agent 实例
    agents = [
        MADDPGAgent(
            agent_id=i,
            obs_dim=OBS_DIM,
            goal_dim=GOAL_DIM,
            act_dim=ACT_DIM,
            n_agents=N_AGENTS,
            latent_dim=LATENT_DIM,
            hidden_dim=HIDDEN_DIM,
            device=DEVICE,
            use_encoder=use_encoder
        )
        for i in range(N_AGENTS)
    ]

    # 判别器模块（可选）
    gail_disc = GAILDiscriminator(OBS_DIM, ACT_DIM, HIDDEN_DIM, device=DEVICE) if use_gail else None

    import os  # Ensure the os module is imported

    # Buffer & Trainer
    buffer = ReplayBuffer(
        buffer_size=100_000,
        obs_dim=OBS_DIM + GOAL_DIM,
        goal_dim=GOAL_DIM,
        act_dim=ACT_DIM,
        n_agents=N_AGENTS,
        device=DEVICE
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
        device=DEVICE,
        obs_dim=OBS_DIM,
        goal_dim=GOAL_DIM,
        tensorboard_logdir=writer_dir  # 传递 TensorBoard 日志路径
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
