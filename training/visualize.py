import imageio
import os

def render_expert_gif(env, policy_fn, gif_path="results/expert_behavior.gif"):
    frames = []
    obs_dict, _ = env.reset()

    for _ in range(env.max_cycles):
        act_dim = env.action_space(env.agents[0]).shape[0]
        actions = {agent: policy_fn(obs_dict[agent], act_dim) for agent in env.agents}

        obs_next, rewards, terminations, truncations, _ = env.step(actions)
        obs_dict = obs_next

        frame = env.render()
        if frame is not None:  # 过滤掉无效帧
            frames.append(frame)

        if all(terminations.values()) or all(truncations.values()):
            break

    if not frames:
        raise RuntimeError("没有生成有效的帧数据")

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=0.1)
    print(f"✅ GIF 已保存到: {gif_path}")