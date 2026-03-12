from pettingzoo.mpe.multi_agents.env import parallel_env  # 修改导入路径

def create_env(render=False, use_encoder=False):
    env = parallel_env(
        continuous_actions=True,
        max_cycles=125,
        render_mode="rgb_array" if render else None
    )
    env.reset()

    # ✅ 正确设置 scenario.use_encoder_obs 标志位
    env.unwrapped.scenario.use_encoder_obs = use_encoder

    return env

