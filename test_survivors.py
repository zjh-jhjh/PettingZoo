# -*- coding: utf-8 -*-
import numpy as np
from utils.env_utils import create_env

# Create environment
env = create_env(render=True)
env.reset()

# Get initial positions
world = env.unwrapped.world
survivors_initial_pos = []
for s in world.survivors:
    survivors_initial_pos.append(s.state.p_pos.copy())

print("Initial survivor positions:")
for i, pos in enumerate(survivors_initial_pos):
    print(f"Survivor {i}: {pos}")

# Execute several random actions
for _ in range(10):
    actions = {agent: np.random.uniform(-1, 1, 5) for agent in env.agents}
    env.step(actions)

# Check if survivor positions have changed
print("\nSurvivor positions after 10 steps:")
for i, s in enumerate(world.survivors):
    print(f"Survivor {i}: {s.state.p_pos}")
    print(f"Position changed: {not np.array_equal(s.state.p_pos, survivors_initial_pos[i])}")
    print(f"Velocity: {s.state.p_vel}")