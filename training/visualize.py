import imageio
import os

def render_expert_gif(env, policy_fn, gif_path="results/expert_behavior.gif"):
    frames = []
    obs_dict, _ = env.reset()

    for _ in range(env.unwrapped.max_cycles):  # ✅ 推荐使用 env.unwrapped
        # ✅ 确保只对已有 obs 的 agent 调用 policy
        actions = {
            agent_id: policy_fn(obs_dict[agent_id], agent_id)
            for agent_id in env.agents
            if agent_id in obs_dict
        }

        obs_next, rewards, terminations, truncations, _ = env.step(actions)
        obs_dict = obs_next

        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if all(terminations.values()) or all(truncations.values()):
            break

    if not frames:
        raise RuntimeError("没有生成有效的帧数据")

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=0.1)
    print(f"✅ GIF 已保存到: {gif_path}")
