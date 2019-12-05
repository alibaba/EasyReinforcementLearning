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

import gym
import numpy as np

from easy_rl.agents import agents
from easy_rl.utils.window_stat import WindowStat

MODEL_CONFIG = dict(
    # specific
    type="Vtrace",

    # common
    init_lr=1e-3,
    lr_strategy_spec={
        'type': 'exponential_decay',
        'decay_steps': 100,
        'decay_rate': 0.9
    },
    global_norm_clip=40)

AGENT_CONFIG = dict(
    type="Agent",
    sample_batch_size=128,
    batch_size=128,

    # no need to compute targets
    compute_targets=False)


def main():
    env = gym.make("CartPole-v0")

    agent_class = agents[AGENT_CONFIG["type"]]
    agent = agent_class(
        env.observation_space,
        env.action_space,
        AGENT_CONFIG,
        MODEL_CONFIG,
        distributed_spec={},
        checkpoint_dir="ckpt_dir",
        export_dir="dump_dir")

    reward_window = WindowStat("reward", 50)
    length_window = WindowStat("length", 50)
    loss_window = WindowStat("loss", 50)
    obs, actions, rewards, dones, next_obs, logits = list(), list(), list(
    ), list(), list(), list()
    act_count = 0

    for i in range(500):
        ob = env.reset()
        done = False
        episode_reward = .0
        episode_len = 0

        while not done:
            action, results = agent.act([ob], False)
            next_ob, reward, done, info = env.step(action[0])
            act_count += 1

            obs.append(ob)
            actions.append(action[0])
            rewards.append(reward)
            dones.append(done)
            next_obs.append(next_ob)
            logits.append(results["logits"][0])
            if agent.ready_to_send:
                agent.send_experience(
                    obs=obs,
                    actions=actions,
                    rewards=rewards,
                    next_obs=next_obs,
                    dones=dones,
                    logits=logits)
            if agent.ready_to_receive:
                batch_data = agent.receive_experience()
                res = agent.learn(batch_data)
                loss_window.push(res["loss"])

            ob = next_ob
            episode_reward += reward
            episode_len += 1
            if act_count % 1000 == 0:
                print("timestep:", act_count, reward_window, length_window)

        reward_window.push(episode_reward)
        length_window.push(episode_len)

    agent.export_saved_model()
    print("Done.")


if __name__ == "__main__":
    main()
