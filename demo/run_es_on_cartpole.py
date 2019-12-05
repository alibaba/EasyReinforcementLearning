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
import numpy as np
import gym

from easy_rl.agents import agents
from easy_rl.models import EvolutionStrategy
from easy_rl.utils.window_stat import WindowStat

MODEL_CONFIG = dict(
    # specific
    type="ES",

    # common
    init_lr=0.01,
    lr_strategy_spec={
        'type': 'exponential_decay',
        'decay_steps': 50,
        'decay_rate': 0.9
    },
    global_norm_clip=40)

AGENT_CONFIG = dict(
    type="Agent",
    # how many trials needed for one optimization
    sample_batch_size=100,
)
np.random.seed(0)


class MyESmodel(EvolutionStrategy):
    def _encode_obs(self, input_obs, scope="encode_obs"):
        with tf.variable_scope(name_or_scope=scope):
            h1 = tf.layers.dense(
                input_obs,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.1, seed=0))
            h2 = tf.layers.dense(
                h1,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.1, seed=0))
            logits = tf.layers.dense(
                h2,
                units=2,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.1, seed=0))
            return logits


def main():
    env = gym.make("CartPole-v0")
    env.seed(0)

    agent_class = agents[AGENT_CONFIG["type"]]
    agent = agent_class(
        env.observation_space,
        env.action_space,
        AGENT_CONFIG,
        MODEL_CONFIG,
        distributed_spec={},
        checkpoint_dir="ckpt_dir",
        custom_model=MyESmodel)

    reward_window = WindowStat("reward", 50)
    length_window = WindowStat("length", 50)

    init_perturbation_scale = 0.1

    seeds, rewards, perturbation_scales = list(), list(), list()
    is_positive_direction = list()

    # how many episodes needed for one trial
    episode_per_perturbation = 1

    returns = list()

    for i in range(4000):
        ob = env.reset()
        done = False
        episode_reward = .0
        episode_len = 0

        if i % episode_per_perturbation == 0:
            # perturb parameters every `episode_per_seed` episodes
            is_positive = True if len(
                is_positive_direction
            ) == 0 else is_positive_direction[-1] != True

            # each seed twice
            seed = np.random.randint(1000000) if is_positive else seeds[-1]
            perturbation_scale = max(
                init_perturbation_scale * (1 - i / 2000.0), 0.02)

            feed = agent.model.perturbation_feed
            fetch = [agent.model.reset_perturbation_op]

            agent.executor.run(
                fetches=fetch,
                feed_dict={
                    feed['perturbation_seeds']: [seed],
                    feed['perturbation_scales']: [perturbation_scale],
                    feed['positive_perturbation']: is_positive
                })

            if is_positive:
                seeds.append(seed)
                perturbation_scales.append(perturbation_scale)
            is_positive_direction.append(is_positive)

        while not done:
            action, result = agent.act([ob], True, use_perturbed_action=True)

            next_ob, reward, done, info = env.step(action[0])

            ob = next_ob
            episode_reward += reward
            episode_len += 1

        rewards.append(episode_reward)
        reward_window.push(episode_reward)
        length_window.push(episode_len)
        if len(rewards) == episode_per_perturbation:
            returns.append(np.mean(rewards))
            rewards = []
            if len(returns) == 2 * agent.config.get('sample_batch_size', 100):
                print(reward_window)
                assert len(seeds) == (len(returns) / 2)
                assert len(perturbation_scales) == (len(returns) / 2)
                agent.learn(
                    batch_data=dict(
                        perturbation_seeds=seeds,
                        perturbation_scales=perturbation_scales,
                        returns=np.reshape(returns, [-1, 2])))
                seeds = []
                perturbation_scales = []
                returns = []
                is_positive_direction = []

                # evaluation 20 episodes
                test_rewards = list()
                for j in range(20):
                    done = False
                    ob = env.reset()
                    episode_reward = 0
                    episode_len = 0
                    while not done:
                        action, result = agent.act(
                            [ob], True, use_perturbed_action=False)

                        next_ob, reward, done, info = env.step(action[0])

                        ob = next_ob
                        episode_reward += reward
                        episode_len += 1
                    test_rewards.append(episode_reward)
                print("[evaluation] average reward of 20 episodes:",
                      np.mean(test_rewards))

                print('train at ', i)

    agent.export_saved_model(export_dir="dump_dir")
    print("Done.")


if __name__ == "__main__":
    main()
