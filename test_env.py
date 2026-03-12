import os
import sys

# 添加项目根目录到 Python 模块搜索路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from pettingzoo.mpe.multi_agents.env import parallel_env

# 创建环境
env = parallel_env(
    continuous_actions=True,
    max_cycles=125,
    render_mode="rgb_array"
)

# 重置环境
obs_dict, _ = env.reset()

# 打印环境信息
print("Agents:", env.agents)
print("Observation space:", env.observation_space(env.agents[0]))
print("Action space:", env.action_space(env.agents[0]))

# 检查 world.survivors
world = env.unwrapped.world
print("World survivors:", len(world.survivors))
for i, s in enumerate(world.survivors):
    print(f"Survivor {i} position:", s.state.p_pos)
    print(f"Survivor {i} movable:", s.movable)

# 执行一步
actions = {agent: env.action_space(agent).sample() for agent in env.agents}
obs_dict, rewards, terminations, truncations, infos = env.step(actions)

# 检查 survivors 是否移动
print("\nAfter step:")
for i, s in enumerate(world.survivors):
    print(f"Survivor {i} position:", s.state.p_pos)