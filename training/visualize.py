import imageio
import os

import os
import imageio


def render_expert_gif(env, policy_fn, gif_path="results/expert_behavior.gif"):
    """
    使用 expert policy 在新的 multi_agents 环境中生成 GIF 可视化
    """
    frames = []

    obs_dict, _ = env.reset()

    for _ in range(env.unwrapped.max_cycles):
        # ✅ 根据当前 obs_dict 计算动作
        actions = {}
        for agent_id in env.agents:
            if agent_id in obs_dict:
                actions[agent_id] = policy_fn(obs_dict[agent_id], agent_id)

        # ✅ 执行一步交互
        obs_next, rewards, terminations, truncations, _ = env.step(actions)
        obs_dict = obs_next

        # ✅ 获取渲染帧
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        # ✅ 检查是否全部结束
        if all(terminations.values()) or all(truncations.values()):
            break

    if not frames:
        raise RuntimeError("没有生成有效的帧数据")

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=0.1)
    print(f"✅ GIF 已保存到: {gif_path}")
