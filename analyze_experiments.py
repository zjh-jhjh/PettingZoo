import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# 各方法的 TensorBoard 路径
log_paths = {
    "A_MADDPG": "logs/runs/maddpg_baseline",
    "B_MADDPG+GAIL": "logs/runs/maddpg_gail",
    "C_MADDPG+Encoder": "logs/runs/maddpg_encoder",
    "D_Full": "logs/runs/maddpg_full"
}

def extract_rewards(path, agent_id=0):
    ea = event_accumulator.EventAccumulator(path)
    ea.Reload()
    tag = f"eval/reward_agent_{agent_id}"
    if tag not in ea.Tags()["scalars"]:
        return [], []
    steps = []
    rewards = []
    for e in ea.Scalars(tag):
        steps.append(e.step)
        rewards.append(e.value)
    return steps, rewards

def plot_rewards(log_paths):
    plt.figure(figsize=(10, 6))
    for label, path in log_paths.items():
        if not os.path.exists(path):
            print(f"⚠️ Skip missing path: {path}")
            continue
        steps, rewards = extract_rewards(path)
        if not steps:
            continue
        plt.plot(steps, rewards, label=label)

    plt.title("Average Episode Reward per Method")
    plt.xlabel("Training Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/reward_comparison.png")
    plt.show()

if __name__ == "__main__":
    plot_rewards(log_paths)
