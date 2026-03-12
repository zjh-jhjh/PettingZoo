#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from tensorboard.backend.event_processing import event_accumulator

def check_tensorboard_tags(log_path):
    """
    检查TensorBoard日志中的可用标签
    """
    try:
        ea = event_accumulator.EventAccumulator(log_path)
        ea.Reload()
        
        print(f"\n=== 检查日志路径: {log_path} ===")
        
        # 获取所有标量标签
        scalar_tags = ea.Tags()["scalars"]
        print(f"\n可用的标量标签 ({len(scalar_tags)} 个):")
        for i, tag in enumerate(scalar_tags, 1):
            print(f"  {i:2d}. {tag}")
        
        # 检查一些常见的标签
        common_tags = [
            "eval/total_episode_reward",
            "eval/episode_reward", 
            "train/episode_reward",
            "reward",
            "episode_reward",
            "total_reward"
        ]
        
        print("\n检查常见标签:")
        for tag in common_tags:
            if tag in scalar_tags:
                events = ea.Scalars(tag)
                print(f"  ✅ {tag}: {len(events)} 个数据点")
                if len(events) > 0:
                    print(f"     - 步数范围: {events[0].step} ~ {events[-1].step}")
                    print(f"     - 值范围: {min(e.value for e in events):.3f} ~ {max(e.value for e in events):.3f}")
            else:
                print(f"  ❌ {tag}: 不存在")
        
        # 显示前几个数据点的详细信息
        if scalar_tags:
            first_tag = scalar_tags[0]
            events = ea.Scalars(first_tag)
            print(f"\n标签 '{first_tag}' 的前5个数据点:")
            for i, event in enumerate(events[:5]):
                print(f"  {i+1}. step={event.step}, value={event.value:.6f}, wall_time={event.wall_time}")
                
    except Exception as e:
        print(f"❌ 处理 {log_path} 时出错: {e}")

if __name__ == "__main__":
    # 检查所有方法的日志
    log_paths = {
        "baseline": "logs/runs/maddpg_baseline",
        "gail": "logs/runs/maddpg_gail", 
        "encoder": "logs/runs/maddpg_encoder",
        "full": "logs/runs/maddpg_full"
    }
    
    for method, path in log_paths.items():
        if os.path.exists(path):
            print(f"\n{'='*60}")
            print(f"检查方法: {method.upper()}")
            check_tensorboard_tags(path)
        else:
            print(f"\n❌ 路径不存在: {path}")