import os
from typing import Callable, Optional

import numpy as np
import torch
from gymnasium.spaces import Box
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    """
    wrapper 环境构造器（与 PettingZoo 的 simple_* 风格保持一致）
    """
    def __init__(
        self,
        N=2,
        max_cycles=25,
        continuous_actions=True,
        render_mode=None,
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
        self.metadata["name"] = "simple_rescue_v0"
        # ✅ 覆盖动作空间为 [-1, 1]
        if self.agents and len(self.agents) > 0:  # 确保 agents 列表非空
            act_dim = self.action_space(self.agents[0]).shape[0]  # 动作维度
            self.action_spaces = {
                agent: Box(low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32)
                for agent in self.possible_agents
            }


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    """
    改造后的 Scenario：多无人机灾后搜救（协作），兼容 MADDPG + GAIL + Encoder。
    - 不考虑 adversary（所有 agents 都为搜救无人机）
    - 使用 world.landmarks 表示 survivors（每个 landmark 增加 .found）
    - use_encoder_obs: 如果 True，observation 不显式包含目标相对位置信息（由 encoder 决定）
                       如果 False，observation 会把 goal_rel_pos 拼进去（raw 模式）
    - use_gail / gail_reward_callback: 若启用 GAIL，会优先使用判别器给出的 pseudo-reward
    - mode: "baseline" | "gail" | "encoder" | "full"  （可在 trainer 中设置）
    """
    def __init__(self):
        self.use_gail = False
        self.use_encoder_obs = False
        self.mode = "baseline"  # 可取 "baseline" | "gail" | "encoder" | "full"
        self.gail_reward_callback: Optional[Callable[[object, object], float]] = self._gail_reward
        self.gail_discriminator = None

    def make_world(self, N=2):
        world = World()
        world.dim_c = 2  # 通信维度

        # N 个 good agents（无人机）
        num_agents = N
        num_survivors = N
        world.num_agents = num_agents

        # ✅ 初始化 agents
        world.agents = [Agent() for _ in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.adversary = False
            agent.name = f"agent_{i}"
            agent.collide = False
            agent.silent = True
            agent.size = 0.15

            # 绑定目标（初始化为 None，reset 时分配）
            agent.goal = None
            agent.goal_id = -1  # 初始化为 -1 而不是 None，与 Agent 类定义一致

        # ✅ 初始化 survivors（灾后幸存者）
        world.survivors = [Landmark() for _ in range(num_survivors)]
        for s in world.survivors:
            s.collide = False
            s.movable = False  # 🚨 保证 survivors 永远不会被更新位置
            s.size = 0.05
            s.color = np.array([0.0, 1.0, 0.0])  # 绿色
            s.found = False

        # ✅ 将 survivors 放入 world.landmarks 中，确保它们被正确处理
        world.landmarks = world.survivors.copy()

        # 目标分配：一一对应（也可在 reset 时打乱）
        for i, agent in enumerate(world.agents):
            if i < len(world.survivors):
                agent.goal = world.survivors[i]
                agent.goal_id = i

        return world


    def reset_world(self, world, np_random):
        """
        初始化 agents 与 survivors 的位置。
        注意 survivors 与 landmarks 分离，不受 world.step() 动力学影响。
        """
        # ✅ 重置 agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])  # 蓝色
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # ✅ 重置 survivors（绿色/深绿色）
        for i, s in enumerate(world.survivors):
            s.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            s.state.p_vel = np.zeros(world.dim_p)  # 🚨 永远保持 0
            s.found = False
            s.color = np.array([0.0, 1.0, 0.0])  # 默认绿色

        # 分配目标（可在这里重新分配）
        for i, agent in enumerate(world.agents):
            if i < len(world.survivors):
                agent.goal = world.survivors[i]
                agent.goal_id = i
                # ✅ 目标 survivor 标记为深绿色
                agent.goal.color = np.array([0.15, 0.65, 0.15])


    def benchmark_data(self, agent, world):
        """
        给出用于 benchmark 的数据（可选）
        """
        # 返回与所有 survivor 的距离（用于可视化/诊断）
        dists = [np.linalg.norm(agent.state.p_pos - lm.state.p_pos) for lm in world.landmarks]
        return tuple(dists)

    def reward(self, agent, world):
        """
        动态 reward：如果启用了 GAIL（use_gail 或 mode == 'gail'/'full'），优先使用 gail_reward_callback。
        否则使用内建的 shaped reward（负距离 + 发现奖励）。
        """
        # 若开启 GAIL 模式且判别器回调可用，优先使用
        if (self.use_gail or self.mode == "gail" or self.mode == "full") and self.gail_reward_callback is not None:
            try:
                rew = self.gail_reward_callback(agent, world)
                # 如果判别器返回 None 或非法，则 fallback 到 shaped reward
                if rew is None:
                    return self._shaped_agent_reward(agent, world)
                try:
                    # 处理数值非法的情况（例如 NaN/Inf）
                    if not np.isfinite(float(rew)):
                        return self._shaped_agent_reward(agent, world)
                except Exception:
                    return self._shaped_agent_reward(agent, world)
                return float(rew)
            except Exception:
                # 若 GAIL 出错，降级为 shaped reward
                return self._shaped_agent_reward(agent, world)

        # 默认 shaped reward
        return self._shaped_agent_reward(agent, world)

    def _shaped_agent_reward(self, agent, world):
        """
        shaped reward:
         - 如果 agent 与其 assigned survivor 距离小于阈值且 survivor 未被发现：标记 found 并给一次性发现奖励
         - 否则按距离给负值（靠近越少惩罚），用于加速学习
        """
        if agent.goal is None:
            return 0.0

        discovery_thresh = 0.1
        rew = 0.0

        # 发现奖励（一次性）
        if not agent.goal.found:
            dist_to_goal = np.linalg.norm(agent.state.p_pos - agent.goal.state.p_pos)
            if dist_to_goal < discovery_thresh:
                agent.goal.found = True
                agent.goal.color = np.array([0.6, 0.6, 0.6])  # 已发现变灰色
                rew += 10.0  # 发现奖励
            else:
                # shaping：鼓励靠近
                rew += -0.5 * dist_to_goal
        else:
            # 如果已被发现，鼓励去帮助其它未被发现的 survivor（可以为 0）
            # 为了简单起见，给 small shaping toward nearest un-found survivor
            unfound = [lm for lm in world.landmarks if not lm.found]
            if unfound:
                dists = [np.linalg.norm(agent.state.p_pos - lm.state.p_pos) for lm in unfound]
                rew += -0.2 * min(dists)
            else:
                # 所有已被发现，给小奖励（鼓励停留/节能）
                rew += 0.0
        return float(rew)

    def observation(self, agent, world):
        """
        观测函数：
        - 自身位置、速度 (2 + 2)
        - 对于每个 survivor (landmark)：若在感知范围内且未 found，则返回带噪声的相对位置；否则返回 0 向量
        - 对于其他 agents：返回相对位置
        - raw 模式 (use_encoder_obs=False)：在 obs 最前拼接 assigned goal 的相对位置
        - encoder 模式 (use_encoder_obs=True)：不拼接 goal，相对位置信息由 encoder 自行处理
        """
        sensor_range = 1.0
        obs_parts = []

        # 自身位置 + 速度
        obs_parts.append(agent.state.p_pos.copy())   # shape (2,)
        obs_parts.append(agent.state.p_vel.copy())   # shape (2,)

        # survivors 相对位置
        for lm in world.survivors:
            rel = lm.state.p_pos - agent.state.p_pos
            dist = np.linalg.norm(rel)
            if dist < sensor_range and not lm.found:
                noisy_rel = rel + np.random.normal(0.0, 0.02, size=rel.shape)
                obs_parts.append(noisy_rel)
            else:
                obs_parts.append(np.zeros_like(rel))

        # 其他 agents 的相对位置
        for other in world.agents:
            if other is agent:
                continue
            obs_parts.append(other.state.p_pos - agent.state.p_pos)

        # 最终拼接
        if not agent.adversary and not self.use_encoder_obs and agent.goal is not None:
            # raw 模式：拼接 goal 相对位置到最前
            goal_rel = agent.goal.state.p_pos - agent.state.p_pos
            obs = np.concatenate([goal_rel] + obs_parts)
        else:
            # encoder 模式：不拼 goal
            obs = np.concatenate(obs_parts)

        return obs.astype(np.float32)


    # GAIL 伪 reward 回调实现（会被 trainer 的判别器注入到 scenario.gail_discriminator）
    def _gail_reward(self, agent, world) -> float | None:
        if self.gail_discriminator is None:
            # 无判别器时返回 None，让外层回退到 shaped reward
            return None

        # 从 observation() 获取当前 obs（注意：这里用 scenario 的 observation）
        obs_np = self.observation(agent, world)
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)

        # 从 agent.action 取得当前动作（PettingZoo 的 Agent.action.u 通常存储在 step 时）
        act = getattr(agent.action, "u", None)
        if act is None:
            # 动作不可用时返回 None，让外层回退
            return None
        act_tensor = torch.tensor(np.array(act, dtype=np.float32)).unsqueeze(0)

        # 使用判别器计算伪 reward（假设判别器实现 compute_reward(obs, act)）
        with torch.no_grad():
            rew = self.gail_discriminator.compute_reward(obs_tensor, act_tensor)
        try:
            val = float(rew.item() if hasattr(rew, "item") else rew)
        except Exception:
            return None
        # 数值非法时回退
        if not np.isfinite(val):
            return None
        return val

    # 兼容旧接口：返回所有非 adversary 的 agents（现在全部是 good agents）
    def good_agents(self, world):
        return [a for a in world.agents if not getattr(a, "adversary", False)]

    # 保持接口一致（不再有 adversary）
    def adversaries(self, world):
        return []
