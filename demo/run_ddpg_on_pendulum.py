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
import time

from easy_rl.agents import agents
from easy_rl.utils.window_stat import WindowStat
from easy_rl.models import DDPGModel

MODEL_CONFIG = dict(
    # specific
    type="DDPG",

    # common
    parameter_noise=False,  # set True to use parameter_noise
    gamma=0.99,
    actor_lr_init=1e-2,
    actor_lr_strategy_spec={
        'type': 'polynomial_decay',
        'decay_steps': 10000,
        'end_learning_rate': 1e-4
    },
    critic_lr_init=1e-2,
    critic_lr_strategy_spec={
        'type': 'polynomial_decay',
        'decay_steps': 13000,
        'end_learning_rate': 1e-3
    },
    global_norm_clip=100,
    ornstein_uhlenbeck_spec={
        "sigma": 0.1,
        "theta": 0.3,
        "noise_scale": 1.0
    },
)

AGENT_CONFIG = dict(
    type="Agent",
    sample_batch_size=8,
    buffer_size=50000,
    learning_starts=2000,
    prioritized_replay=False,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta=0.4,
    batch_size=1024,
    sync_target_frequency=200,
    perturbation_frequency=50,  # recommend to set to 50
    noise_kl_episodes=1000  # 1000 episode kl_threshold will decay to 1e-4
)
np.random.seed(0)


class MyDDPG(DDPGModel):
    def _encode_obs(self, input_obs, scope="encode_obs"):
        with tf.variable_scope(name_or_scope=scope):
            h1 = tf.layers.dense(
                input_obs,
                units=32,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.1, seed=0))
            h2 = tf.layers.dense(
                h1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.1, seed=0))

            return h2

    def _encode_obs_action(self,
                           input_obs,
                           input_action,
                           scope="encode_obs_action"):
        with tf.variable_scope(name_or_scope=scope):
            state_emb = tf.layers.dense(
                input_obs,
                units=32,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.1, seed=0))
            state_emb_action = tf.concat([state_emb, input_action], axis=1)
            h1 = tf.layers.dense(
                state_emb_action,
                units=16,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.1, seed=0))
            h2 = tf.layers.dense(
                h1,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.1, seed=0))

            return tf.squeeze(h2)


def main():
    env = gym.make("Pendulum-v0")
    env.seed(0)

    agent_class = agents[AGENT_CONFIG["type"]]
    agent = agent_class(
        env.observation_space,
        env.action_space,
        AGENT_CONFIG,
        MODEL_CONFIG,
        distributed_spec={},
        checkpoint_dir="ckpt_dir",
        export_dir="dump_dir",
        custom_model=MyDDPG)

    reward_window = WindowStat("reward", 50)
    length_window = WindowStat("length", 50)
    loss_window = WindowStat("loss", 50)
    actor_loss = WindowStat("actor_loss", 50)
    obs, actions, rewards, next_obs, dones = list(), list(), list(), list(
    ), list()
    act_count = 0
    train_count = 0
    total_cost = time.time()
    for i in range(500):
        ob = env.reset()
        done = False
        episode_reward = .0
        episode_len = 0

        while not done:
            action, results = agent.act(
                [ob], False, use_perturbed_action=False)
            act_count += 1
            next_ob, reward, done, info = env.step(action[0])
            obs.append(ob)
            actions.append(action[0])
            rewards.append(0.1 * reward)
            next_obs.append(next_ob)
            dones.append(done)
            if agent.ready_to_send:
                agent.send_experience(
                    obs=obs,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    next_obs=next_obs)
            if agent.ready_to_receive:
                batch_data = agent.receive_experience()
                res = agent.learn(batch_data)
                loss_window.push(res["critic_loss"])
                actor_loss.push(res["actor_loss"])
                train_count += 1

                if AGENT_CONFIG.get("prioritized_replay", False):
                    agent.update_priorities(
                        indexes=batch_data["indexes"],
                        td_error=res["td_error"])

            ob = next_ob
            episode_reward += reward
            episode_len += 1
        agent.add_episode(1)
        reward_window.push(episode_reward)
        length_window.push(episode_len)
        if act_count % 200 == 0:
            print("timestep:", act_count, reward_window, loss_window,
                  actor_loss)

    agent.export_saved_model()
    print("Done.", "act_count:", act_count, "train_count:", train_count,
          "total_cost:",
          time.time() - total_cost)


if __name__ == "__main__":
    main()
