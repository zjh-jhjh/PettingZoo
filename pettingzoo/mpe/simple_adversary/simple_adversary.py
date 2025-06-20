# noqa: D212, D415
"""
# Simple Adversary

```{figure} mpe_simple_adversary.gif
:width: 140px
:name: simple_adversary
```

```{eval-rst}
.. warning::

    The environment `pettingzoo.mpe.simple_adversary_v3` has been moved to the new `MPE2 package <https://mpe2.farama.org>`_, and will be removed from PettingZoo in a future release.
    Please update your import to `mpe2.simple_adversary_v3`.

```

This environment is part of the <a href='..'>MPE environments</a>. Please read that page first for general information.

| Import             | `from pettingzoo.mpe import simple_adversary_v3` |
|--------------------|--------------------------------------------------|
| Actions            | Discrete/Continuous                              |
| Parallel API       | Yes                                              |
| Manual Control     | No                                               |
| Agents             | `agents= [adversary_0, agent_0,agent_1]`         |
| Agents             | 3                                                |
| Action Shape       | (5)                                              |
| Action Values      | Discrete(5)/Box(0.0, 1.0, (5))                   |
| Observation Shape  | (8),(10)                                         |
| Observation Values | (-inf,inf)                                       |
| State Shape        | (28,)                                            |
| State Values       | (-inf,inf)                                       |


In this environment, there is 1 adversary (red), N good agents (green), N landmarks (default N=2). All agents observe the position of landmarks and other agents. One landmark is the 'target landmark' (colored green). Good agents are rewarded based on how close the closest one of them is to the
target landmark, but negatively rewarded based on how close the adversary is to the target landmark. The adversary is rewarded based on distance to the target, but it doesn't know which landmark is the target landmark. All rewards are unscaled Euclidean distance (see main MPE documentation for
average distance). This means good agents have to learn to 'split up' and cover all landmarks to deceive the adversary.

Agent observation space: `[goal_rel_position, landmark_rel_position, other_agent_rel_positions]`

Adversary observation space: `[landmark_rel_position, other_agents_rel_positions]`

Agent action space: `[no_action, move_left, move_right, move_down, move_up]`

Adversary action space: `[no_action, move_left, move_right, move_down, move_up]`

### Arguments

``` python
simple_adversary_v3.env(N=2, max_cycles=25, continuous_actions=False, dynamic_rescaling=False)
```



`N`:  number of good agents and landmarks

`max_cycles`:  number of frames (a step for each agent) until game terminates

`continuous_actions`: Whether agent action spaces are discrete(default) or continuous

`dynamic_rescaling`: Whether to rescale the size of agents and landmarks based on the screen size

"""

import numpy as np
import torch
from gymnasium.utils import EzPickle
from typing import Callable, Optional

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=2,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None, # "human"  # "rgb_array", "ansi"
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "simple_adversary_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def __init__(self):
        self.use_gail = False
        self.gail_reward_callback: Optional[Callable[[object, object], float]] = self._gail_reward # 实例化后赋值
        self.gail_discriminator = None  # 实例化后赋值
    def make_world(self, N=2):
        world = World()
        # 设置通信维度（2D）
        world.dim_c = 2

        # agent 总数 = adversary 数（固定 1）+ good agents 数（N）
        num_adversaries = 1
        num_good_agents = N
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_good_agents  # 每个 good agent 分配一个目标 landmark

        world.num_agents = num_agents

        # 创建 agent 实体
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = (i < num_adversaries)
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if agent.adversary else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"

            agent.collide = False
            agent.silent = True
            agent.size = 0.15

        # 创建 landmarks
        world.landmarks = [Landmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark_{i}"
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.08

        # 👇 目标分配：将每个 good agent 绑定一个目标 landmark
        good_agents = [a for a in world.agents if not a.adversary]
        for i, agent in enumerate(good_agents):
            # 目标 landmark：默认一一对应（agent_0 对 landmark_0）
            agent.goal = world.landmarks[i]
            agent.goal_id = i  # 可用于记录目标索引

        return world


    def reset_world(self, world, np_random):
        # 设置 adversary agent 的颜色（红色）
        world.agents[0].color = np.array([0.85, 0.35, 0.35])

        # 设置 good agents 的颜色（蓝色）
        for i in range(1, world.num_agents):
            world.agents[i].color = np.array([0.35, 0.35, 0.85])

        # 设置所有 landmark 的默认颜色（灰）
        for landmark in world.landmarks:
            landmark.color = np.array([0.15, 0.15, 0.15])

        # 为每个 good agent 分配一个唯一目标 landmark
        good_agents = [a for a in world.agents if not a.adversary]
        for i, agent in enumerate(good_agents):
            agent.goal = world.landmarks[i]
            agent.goal_id = i

            # 为该 landmark 着色，标记被选为目标（绿色）
            agent.goal.color = np.array([0.15, 0.65, 0.15])

        # 初始化 agent 的位置和状态
        for agent in world.agents:
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # 初始化 landmark 的位置和状态
        for landmark in world.landmarks:
            landmark.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            return np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:
            dists = []
            for lm in world.landmarks:
                dists.append(np.sum(np.square(agent.state.p_pos - lm.state.p_pos)))
            dists.append(
                np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
            )
            return tuple(dists)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # 使用 GAIL reward（如果启用）
        if self.use_gail and self.gail_reward_callback is not None:
            return self.gail_reward_callback(agent, world)

        # 否则使用内建 reward 函数
        return (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )


    def agent_reward(self, agent, world):
        # 若 agent 没有分配目标，默认返回 0
        if agent.goal is None:
            return 0.0

        shaped_reward = True

        if shaped_reward:
            # 奖励为负距离（越近越好）
            dist = np.linalg.norm(agent.state.p_pos - agent.goal.state.p_pos)
            return -dist
        else:
            # 二值奖励（是否进入 radius 范围）
            if np.linalg.norm(agent.state.p_pos - agent.goal.state.p_pos) < agent.goal.size * 2:
                return 5.0
            else:
                return 0.0


    def adversary_reward(self, agent, world):
        shaped_reward = True
        good_agents = self.good_agents(world)

        if shaped_reward:
            # 距离越小，reward 越高（负距离）
            dists = [np.linalg.norm(agent.state.p_pos - a.state.p_pos) for a in good_agents]
            return -min(dists)
        else:
            for a in good_agents:
                if np.linalg.norm(agent.state.p_pos - a.state.p_pos) < agent.size + a.size:
                    return 5.0
            return 0.0


    def observation(self, agent, world):
        # 所有 landmark 的相对位置
        entity_pos = [entity.state.p_pos - agent.state.p_pos for entity in world.landmarks]

        # 所有其他 agent 的相对位置
        other_pos = [
            other.state.p_pos - agent.state.p_pos
            for other in world.agents if other is not agent
        ]

        # 如果是 good agent，加入目标信息
        if not agent.adversary:
            # 🧠 加入目标 landmark 的相对位置（目标导向输入）
            goal_rel_pos = agent.goal.state.p_pos - agent.state.p_pos
            return np.concatenate([goal_rel_pos] + entity_pos + other_pos)

        # 对于 adversary，仍不提供目标信息
        return np.concatenate(entity_pos + other_pos)

        
    # def gail_reward_callback(self, agent, world):
    #     if self.gail_discriminator is None:
    #         return 0.0  # fallback

    #     # 获取当前 agent 的 obs & act
    #     obs = self.observation(agent, world)  # reuse PettingZoo 定义的 obs
    #     obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    #     # 获取 agent 的动作
    #     act = getattr(agent.action, "u", None)
    #     if act is None:
    #         return 0.0
    #     act_tensor = torch.tensor(act, dtype=torch.float32).unsqueeze(0)

    #     # 使用判别器计算伪 reward
    #     with torch.no_grad():
    #         rew = self.gail_discriminator.compute_reward(obs_tensor, act_tensor)
    #     return rew.item()
    def _gail_reward(self, agent, world) -> float:
        if self.gail_discriminator is None:
            return 0.0

        obs = self.observation(agent, world)
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        act = getattr(agent.action, "u", None)
        if act is None:
            return 0.0
        act_tensor = torch.tensor(act, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            rew = self.gail_discriminator.compute_reward(obs_tensor, act_tensor)
        return rew.item()

