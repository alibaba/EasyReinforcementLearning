# Copyright (c) 2019 Alibaba Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import gym
from gym.spaces import Discrete, Box
import numpy as np

from easy_rl.agents import agents
from easy_rl.models import DQNModel
from easy_rl.utils.window_stat import WindowStat

MODEL_CONFIG = dict(
    # specific
    type="LinUCB")

AGENT_CONFIG = dict(
    type="Agent",
    sample_batch_size=1,
    buffer_size=1,
    learning_starts=0,
    prioritized_replay=False,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta=0.4,
    batch_size=1,
    use_gae=False,
    compute_targets=False,
)
np.random.seed(0)


class MaxComponentEnv(gym.Env):
    def __init__(self, num_arms=2):
        self._num_arms = num_arms
        self.observation_space = Box(
            low=np.zeros((num_arms, ), dtype=np.float32),
            high=np.ones((num_arms, ), dtype=np.float32))
        self.action_space = Discrete(num_arms)

    def reset(self, **kwargs):
        self._cur_state = np.random.uniform(0, 1.0, size=(self._num_arms, ))
        return self._cur_state

    def step(self, action):
        reward = self._cur_state[action]
        self._cur_state = np.random.uniform(0, 1.0, size=(self._num_arms, ))
        return self._cur_state, reward, True, {}


def main():
    env = MaxComponentEnv(num_arms=6)

    agent_class = agents[AGENT_CONFIG["type"]]
    agent = agent_class(
        env.observation_space,
        env.action_space,
        AGENT_CONFIG,
        MODEL_CONFIG,
        distributed_spec={},
        export_dir="hook_dump_dir")

    reward_window = WindowStat("reward", 50)
    length_window = WindowStat("length", 50)
    loss_window = WindowStat("loss", 50)
    obs, actions, rewards, next_obs, dones = list(), list(), list(), list(
    ), list()
    act_count = 0

    for i in range(100):
        ob = env.reset()
        done = False
        episode_reward = .0
        episode_len = 0

        while not done:
            action, results = agent.act(
                [ob], deterministic=False, use_perturbed_action=False)

            next_ob, reward, done, info = env.step(action[0])
            act_count += 1

            obs.append(ob)
            actions.append(action[0])
            rewards.append(reward)
            next_obs.append(next_ob)
            dones.append(done)
            if agent.ready_to_send:
                agent.send_experience(
                    obs=obs,
                    actions=actions,
                    rewards=rewards,
                    next_obs=next_obs,
                    dones=dones)
            if agent.ready_to_receive:
                batch_data = agent.receive_experience()
                res = agent.learn(batch_data)
                loss_window.push(res['loss'])

                if AGENT_CONFIG.get("prioritized_replay", False):
                    agent.update_priorities(
                        indexes=batch_data["indexes"],
                        td_error=res["td_error"])

            ob = next_ob
            episode_reward += reward
            episode_len += 1
            if act_count % 5 == 0:
                print("timestep:", act_count, reward_window, length_window)

        agent.add_episode(1)
        reward_window.push(episode_reward)
        length_window.push(episode_len)

    agent.export_saved_model()
    print("Done.")


if __name__ == "__main__":
    main()
