# -*- coding: utf-8 -*-
import os
import sys
import numpy as np

# 允许从当前仓库直接导入包
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pettingzoo.mpe.multi_agents.env import raw_env


def main():
    env = raw_env(N=2, continuous_actions=True, render_mode=None)
    # 随机重置环境
    env.reset()

    # 取第一个 agent
    agent = env.world.agents[0]

    # 开启 GAIL 但不提供判别器，应该自动回退到 shaped reward
    env.scenario.use_gail = True
    env.scenario.mode = "gail"
    env.scenario.gail_discriminator = None

    rew_gail_path = env.scenario.reward(agent, env.world)

    # 显式关闭 GAIL，获得 shaped reward 进行对比
    env.scenario.use_gail = False
    env.scenario.mode = "baseline"
    rew_shaped = env.scenario.reward(agent, env.world)

    print("GAIL path reward (expected shaped fallback):", rew_gail_path)
    print("Direct shaped reward:", rew_shaped)
    # 两者应几乎一致（允许极小数值误差）
    assert np.isclose(rew_gail_path, rew_shaped, atol=1e-6), "GAIL 回退逻辑未生效"
    print("OK: GAIL 回退到 shaped reward 逻辑验证通过")


if __name__ == "__main__":
    main()
