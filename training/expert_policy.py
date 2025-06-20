import numpy as np
import torch

def expert_policy(obs):
    """
    简单的专家策略示例
    
    Args:
        obs (dict): 包含每个智能体观测值的字典
            key: agent_id
            value: torch.Tensor 观测值 shape: (obs_dim,)
    Returns:
        dict: 包含每个智能体动作的字典
            key: agent_id
            value: torch.Tensor 动作向量 shape: (act_dim,)
    """
    actions = {}
    for agent_id, agent_obs in obs.items():
        # 将 numpy 数组转换为 tensor
        if isinstance(agent_obs, np.ndarray):
            agent_obs = torch.from_numpy(agent_obs).float()
        
        # 所有智能体都使用5维动作空间，并确保动作在 [0,1] 范围内
        act_dim = 5
        
        # 生成随机动作，范围在 [0,1] 之间
        if len(agent_obs.shape) > 1:
            actions[agent_id] = torch.rand((agent_obs.shape[0], act_dim))
        else:
            actions[agent_id] = torch.rand(act_dim)
    
    return actions
