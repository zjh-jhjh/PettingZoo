#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot reward curves comparison for different training methods

Usage:
    python plot_rewards.py --methods baseline gail encoder full --smooth 5
"""

import os
import sys
import argparse

# 添加项目根目录到 Python 模块搜索路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from training.visualize_rewards import plot_all_methods, plot_agents_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Plot reward curves comparison for different training methods")
    parser.add_argument("--methods", type=str, nargs="*", 
                        choices=["baseline", "gail", "encoder", "full"],
                        help="Methods to plot, options: baseline, gail, encoder, full")
    parser.add_argument("--n_agents", type=int, default=2,
                        help="Number of agents")
    parser.add_argument("--smooth", type=int, default=100,
                        help="Smoothing window size")
    parser.add_argument("--output", type=str, default="results/reward_comparison.png",
                        help="Path to save the image")
    parser.add_argument("--agents_plot", action="store_true",
                        help="Whether to plot rewards for each agent")
    parser.add_argument("--font", type=str, default="",
                        help="Chart font, leave empty for default font")
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set font for display
    try:
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = [args.font]  # For displaying Chinese characters properly
        plt.rcParams['axes.unicode_minus'] = False  # For displaying minus sign properly
    except Exception as e:
        print(f"Error setting font: {e}, will use default font")
    
    # Plot total rewards comparison for all methods
    plot_all_methods(
        methods=args.methods,
        n_agents=args.n_agents,
        smooth_window=args.smooth,
        save_path=args.output
    )
    
    # If needed, plot rewards comparison for each agent
    if args.agents_plot:
        agents_output = args.output.replace(".png", "_agents.png")
        plot_agents_comparison(
            methods=args.methods,
            n_agents=args.n_agents,
            smooth_window=args.smooth,
            save_path=agents_output
        )


if __name__ == "__main__":
    main()