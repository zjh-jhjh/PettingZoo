# -*- coding: utf-8 -*-
"""
测试幸存者是否不再移动（扩展版）
"""
import numpy as np
from pettingzoo.mpe.multi_agents import env as rescue_env

# 创建环境
env = rescue_env.parallel_env(render_mode=None)
env.reset()

# 记录幸存者的初始位置
initial_positions = []
for survivor in env.unwrapped.world.survivors:
    initial_positions.append(survivor.state.p_pos.copy())

print("初始幸存者位置:")
for i, pos in enumerate(initial_positions):
    print(f"幸存者 {i}: {pos}")

# 执行多轮测试，每轮10步
num_rounds = 5
for round_idx in range(num_rounds):
    print(f"\n===== 第 {round_idx+1} 轮测试 =====")
    
    # 执行10步随机动作
    for step in range(10):
        actions = {agent: np.random.uniform(-1, 1, env.action_space(agent).shape[0]) for agent in env.agents}
        env.step(actions)
    
    # 检查幸存者的位置
    round_positions = []
    round_velocities = []
    for survivor in env.unwrapped.world.survivors:
        round_positions.append(survivor.state.p_pos.copy())
        round_velocities.append(survivor.state.p_vel.copy())
    
    # 验证位置是否保持不变
    all_unchanged = True
    for i, (initial, current) in enumerate(zip(initial_positions, round_positions)):
        position_changed = not np.array_equal(initial, current)
        print(f"幸存者 {i}: 当前位置 {current}, 位置改变: {position_changed}")
        if position_changed:
            all_unchanged = False
    
    # 验证速度是否为零
    all_zero_vel = True
    for i, vel in enumerate(round_velocities):
        is_zero = np.all(vel == 0)
        print(f"幸存者 {i}: 速度 {vel}, 速度为零: {is_zero}")
        if not is_zero:
            all_zero_vel = False
    
    # 总结本轮结果
    if all_unchanged and all_zero_vel:
        print("本轮测试通过: 所有幸存者位置不变且速度为零")
    else:
        print("本轮测试失败: 有幸存者位置改变或速度不为零")

# 最终结论
print("\n===== 最终结论 =====")
final_positions = []
for survivor in env.unwrapped.world.survivors:
    final_positions.append(survivor.state.p_pos.copy())

all_unchanged = True
for i, (initial, final) in enumerate(zip(initial_positions, final_positions)):
    position_changed = not np.array_equal(initial, final)
    print(f"幸存者 {i}: 初始位置 {initial}, 最终位置 {final}, 位置改变: {position_changed}")
    if position_changed:
        all_unchanged = False

if all_unchanged:
    print("\n测试成功: 所有幸存者位置在整个测试过程中保持不变!")
else:
    print("\n测试失败: 有幸存者在测试过程中移动了!")