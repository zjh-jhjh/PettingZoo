import numpy as np

def expert_policy(agent_obs, act_dim=5):
    obs = np.array(agent_obs)
    goal_vec = obs[:2]  # 假设前两维是目标方向
    norm = np.linalg.norm(goal_vec)
    direction = goal_vec / norm if norm > 1e-6 else np.zeros(2)

    action = np.zeros(act_dim)
    action[:2] = (direction + 1.0) / 2.0  # ✅ 映射到 [0, 1]
    return action
