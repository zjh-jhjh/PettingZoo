from pettingzoo.mpe import simple_adversary_v3

def create_env(render=False, n_good=2, max_cycles=25):
    """
    创建 PettingZoo 环境
    Args:
        render (bool): 是否启用渲染
        n_good (int): good agents 的数量
        max_cycles (int): 最大循环次数
    Returns:
        PettingZoo 环境实例
    """
    return simple_adversary_v3.parallel_env(
        N=n_good,
        max_cycles=max_cycles,
        continuous_actions=True,
        render_mode="rgb_array" if render else None
    )