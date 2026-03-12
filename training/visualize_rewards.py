import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import argparse

# 各方法的 TensorBoard 路径
LOG_PATHS = {
    "baseline": os.path.abspath("logs/runs/maddpg_baseline"),
    "gail": os.path.abspath("logs/runs/maddpg_gail"),
    "encoder": os.path.abspath("logs/runs/maddpg_encoder"),
    "full": os.path.abspath("logs/runs/maddpg_full")
}

# 方法显示名称映射
METHOD_NAMES = {
    "baseline": "MADDPG",
    "gail": "MADDPG+GAIL",
    "encoder": "MADDPG+Encoder",
    "full": "MADDPG+GAIL+Encoder"
}

# 方法对应的颜色
METHOD_COLORS = {
    "baseline": "#1f77b4",  # 蓝色
    "gail": "#ff7f0e",      # 橙色
    "encoder": "#2ca02c",   # 绿色
    "full": "#d62728"       # 红色
}


def extract_rewards(path, tag="train/episode_reward", smooth_window=1):
    """
    从TensorBoard日志中提取奖励数据
    
    参数:
        path: TensorBoard日志路径
        tag: 要提取的标签名称
        smooth_window: 平滑窗口大小
        
    返回:
        steps: 步数列表
        rewards: 奖励列表
        rewards_smooth: 平滑后的奖励列表
    """
    try:
        ea = event_accumulator.EventAccumulator(path)
        ea.Reload()
        
        # If the specified tag doesn't exist, try to find available reward-related tags
        available_tags = ea.Tags()["scalars"]
        reward_tags = [t for t in available_tags if "reward" in t.lower()]
        
        # 优先使用训练奖励标签
        preferred_tags = ["train/episode_reward", "eval/total_episode_reward"]
        tag_found = False
        for preferred_tag in preferred_tags:
            if preferred_tag in available_tags:
                if tag != preferred_tag:
                    print(f"⚠️ Tag '{tag}' not found, using '{preferred_tag}' instead")
                tag = preferred_tag
                tag_found = True
                break
        
        if not tag_found:
            if reward_tags:
                print(f"⚠️ Tag '{tag}' not found, using '{reward_tags[0]}' instead")
                tag = reward_tags[0]
            else:
                print(f"⚠️ No reward-related tags found in {path}")
                return [], [], [], []
            
        # Extract data
        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        rewards = [e.value for e in events]
        
        # Apply smoothing
        if smooth_window > 1 and len(rewards) > smooth_window:
            rewards_smooth = np.convolve(rewards, np.ones(smooth_window)/smooth_window, mode='valid')
            # Adjust steps to match the length of smoothed rewards
            steps_smooth = steps[smooth_window-1:]
            return steps, rewards, rewards_smooth, steps_smooth
        else:
            return steps, rewards, rewards, steps
            
    except Exception as e:
        print(f"❌ Error processing {path}: {e}")
        return [], [], [], []


def extract_agent_rewards(path, n_agents=3, smooth_window=5):
    """
    Extract rewards for each agent
    
    Parameters:
        path: TensorBoard log path
        n_agents: Number of agents
        smooth_window: Smoothing window size
        
    Returns:
        agent_data: Dictionary containing reward data for each agent
    """
    agent_data = {}
    
    for i in range(n_agents):
        # 尝试训练时的智能体奖励标签
        tag = f"train/agent_{i}_reward"
        steps, rewards, rewards_smooth, steps_smooth = extract_rewards(path, tag, smooth_window)
        if not steps:  # 如果训练标签不存在，尝试评估标签
            tag = f"eval/agent_{i}_reward"
            steps, rewards, rewards_smooth, steps_smooth = extract_rewards(path, tag, smooth_window)
        
        if steps:  # Only add when data is successfully extracted
            agent_data[i] = {
                "steps": steps,
                "rewards": rewards,
                "rewards_smooth": rewards_smooth,
                "steps_smooth": steps_smooth
            }
    
    # If no agent rewards are found, try to extract total rewards
    if not agent_data:
        steps, rewards, rewards_smooth, steps_smooth = extract_rewards(path)
        if steps:  # Only add when data is successfully extracted
            agent_data["total"] = {
                "steps": steps,
                "rewards": rewards,
                "rewards_smooth": rewards_smooth,
                "steps_smooth": steps_smooth
            }
    
    return agent_data


def plot_all_methods(methods=None, n_agents=3, smooth_window=5, save_path="results/reward_comparison.png"):
    """
    Plot reward curves for all methods
    
    Parameters:
        methods: List of methods to plot, if None, plot all methods
        n_agents: Number of agents
        smooth_window: Smoothing window size
        save_path: Path to save the image
    """
    if methods is None:
        methods = list(LOG_PATHS.keys())
    
    plt.figure(figsize=(12, 8))
    data_plotted = False
    
    for method in methods:
        path = LOG_PATHS.get(method)
        if not path or not os.path.exists(path):
            print(f"⚠️ Skipping non-existent path: {path}")
            continue
        
        # Extract total rewards
        steps, rewards, rewards_smooth, steps_smooth = extract_rewards(path, smooth_window=smooth_window)
        
        if not steps:
            print(f"⚠️ No data extracted from {method}")
            continue
        
        # Plot smoothed reward curves
        label = METHOD_NAMES.get(method, method)
        color = METHOD_COLORS.get(method, None)
        
        plt.plot(steps_smooth, rewards_smooth, label=label, color=color, linewidth=2)
        # Plot original data points (semi-transparent)
        plt.scatter(steps, rewards, color=color, alpha=0.2, s=10)
        
        data_plotted = True
    
    if not data_plotted:
        print("No data was plotted. Please check your log paths and data.")
        return
    
    plt.title("Training Reward Convergence Comparison", fontsize=16)
    plt.xlabel("Episodes", fontsize=14)
    plt.ylabel("Average Reward", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ Image saved to {save_path}")
    plt.show()


def plot_agents_comparison(methods=None, n_agents=3, smooth_window=5, save_path="results/agents_comparison.png"):
    """
    Plot reward comparison for each agent under different methods
    
    Parameters:
        methods: List of methods to plot, if None, plot all methods
        n_agents: Number of agents
        smooth_window: Smoothing window size
        save_path: Path to save the image
    """
    if methods is None:
        methods = list(LOG_PATHS.keys())
    
    # 创建子图网格
    fig, axes = plt.subplots(len(methods), 1, figsize=(12, 4*len(methods)), sharex=True)
    if len(methods) == 1:
        axes = [axes]  # Ensure axes is always a list
    
    data_plotted = False
    
    for i, method in enumerate(methods):
        ax = axes[i]
        path = LOG_PATHS.get(method)
        if not path or not os.path.exists(path):
            ax.text(0.5, 0.5, f"⚠️ Path does not exist: {path}", ha='center', va='center')
            continue
        
        # Extract rewards for each agent
        agent_data = extract_agent_rewards(path, n_agents, smooth_window)
        
        if not agent_data:
            ax.text(0.5, 0.5, f"⚠️ No data extracted", ha='center', va='center')
            continue
        
        # Plot rewards for each agent
        for agent_id, data in agent_data.items():
            if agent_id == "total":
                label = "Total Reward"
                linestyle = "-"
            else:
                label = f"Agent {agent_id}"
                linestyle = "--" if agent_id > 0 else "-"
            
            ax.plot(data["steps_smooth"], data["rewards_smooth"], 
                   label=label, linestyle=linestyle, linewidth=2)
            ax.scatter(data["steps"], data["rewards"], alpha=0.2, s=10)
        
        ax.set_title(f"Agent Training Rewards for {METHOD_NAMES.get(method, method)}", fontsize=14)
        ax.set_ylabel("Reward", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        data_plotted = True
    
    if data_plotted:
        axes[-1].set_xlabel("Episodes", fontsize=12)
        plt.tight_layout()
        
        # Ensure save directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"✅ Image saved to {save_path}")
        plt.show()
    else:
        plt.close(fig)
        print("No data was plotted. Please check your log paths and data.")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize reward curves for different training methods")
    parser.add_argument("--methods", type=str, nargs="*", choices=LOG_PATHS.keys(),
                        help="Methods to plot, options: baseline, gail, encoder, full")
    parser.add_argument("--n_agents", type=int, default=2,
                        help="Number of agents")
    parser.add_argument("--smooth", type=int, default=5,
                        help="Smoothing window size")
    parser.add_argument("--output", type=str, default="results/reward_comparison.png",
                        help="Path to save the image")
    parser.add_argument("--agents_plot", action="store_true",
                        help="Whether to plot reward comparison for each agent")
    return parser.parse_args()

def summarize_results(methods=None, tag="train/episode_reward", output_path="results/summary.csv"):
    """
    汇总不同方法的训练结果（均值、方差、最大值、收敛趋势）

    参数:
        methods: 要分析的方法列表
        tag: TensorBoard 奖励标签
        output_path: 输出CSV文件路径
    """
    if methods is None:
        methods = list(LOG_PATHS.keys())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    summary_data = []

    print("\n📊 ==== 实验结果统计汇总 ====\n")
    print(f"{'Method':<20}{'Mean(Last1000)':<20}{'Std(Last1000)':<20}{'Max Reward':<20}{'Episodes':<10}")
    print("-" * 90)

    for method in methods:
        path = LOG_PATHS.get(method)
        if not path or not os.path.exists(path):
            print(f"⚠️ 跳过不存在的路径: {path}")
            continue

        # 提取奖励数据
        steps, rewards, rewards_smooth, steps_smooth = extract_rewards(path, tag)
        if not rewards:
            print(f"⚠️ 未从 {method} 提取到数据")
            continue

        rewards = np.array(rewards)
        if len(rewards) >= 1000:
            last_rewards = rewards[-1000:]
        else:
            last_rewards = rewards

        mean_last = np.mean(last_rewards)
        std_last = np.std(last_rewards)
        max_reward = np.max(rewards)

        print(f"{METHOD_NAMES.get(method, method):<20}{mean_last:<20.3f}{std_last:<20.3f}{max_reward:<20.3f}{len(rewards):<10}")

        summary_data.append({
            "Method": METHOD_NAMES.get(method, method),
            "Mean_Last1000": round(mean_last, 3),
            "Std_Last1000": round(std_last, 3),
            "Max_Reward": round(max_reward, 3),
            "Episodes": len(rewards)
        })

    # 保存CSV
    import pandas as pd
    df = pd.DataFrame(summary_data)
    df.to_csv(output_path, index=False)
    print(f"\n✅ 统计结果已保存到 {output_path}\n")


def main():
    args = parse_args()
    
    # Plot total reward comparison for all methods
    plot_all_methods(
        methods=args.methods,
        n_agents=args.n_agents,
        smooth_window=args.smooth,
        save_path=args.output
    )
    
    # If needed, plot reward comparison for each agent
    if args.agents_plot:
        agents_output = args.output.replace(".png", "_agents.png")
        plot_agents_comparison(
            methods=args.methods,
            n_agents=args.n_agents,
            smooth_window=args.smooth,
            save_path=agents_output
        )

    # === 新增：生成训练结果统计表 ===
    summarize_results(methods=args.methods)


if __name__ == "__main__":
    main()