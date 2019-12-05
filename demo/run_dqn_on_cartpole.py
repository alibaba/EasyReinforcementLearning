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
import numpy as np

from easy_rl.agents import agents
from easy_rl.models import DQNModel
from easy_rl.utils.window_stat import WindowStat

MODEL_CONFIG = dict(
    # specific
    type="DQN",
    n_step=3,
    dueling=False,
    double_q=True,
    num_atoms=11,  # recommend to set 11 to run distributional dqn
    v_min=0,
    v_max=25,

    # common
    parameter_noise=False,  # set True to use parameter_noise
    gamma=0.95,
    init_lr=1e-3,
    lr_strategy_spec={
        'type': 'exponential_decay',
        'decay_steps': 1000,
        'decay_rate': 0.9
    },
    global_norm_clip=40)

AGENT_CONFIG = dict(
    type="Agent",
    sample_batch_size=4,
    buffer_size=50000,
    learning_starts=1000,
    prioritized_replay=False,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta=0.4,
    batch_size=256,
    sync_target_frequency=100,
    exploration_timesteps=40000,
    perturbation_frequency=40,  # recommend to set to 50
    noise_kl_episodes=300  # after 300 episodes kl_threshold will decay to 1e-4
)
np.random.seed(0)


class MyDQNModel(DQNModel):
    def _encode_obs(self, input_obs, scope="encode_obs"):
        with tf.variable_scope(name_or_scope=scope):
            h1 = tf.layers.dense(
                input_obs,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=0))
            h2 = tf.layers.dense(
                h1,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=0))
            logits = tf.layers.dense(
                h2,
                units=2 * MODEL_CONFIG["num_atoms"],
                activation=None,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=0))
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
        export_dir="dump_dir",
        checkpoint_dir="ckpt_dir",
        custom_model=MyDQNModel)

    reward_window = WindowStat("reward", 50)
    length_window = WindowStat("length", 50)
    loss_window = WindowStat("loss", 50)
    obs, actions, rewards, next_obs, dones = list(), list(), list(), list(
    ), list()
    act_count = 0

    for i in range(600):
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
            if act_count % 1000 == 0:
                print("timestep:", act_count, reward_window, length_window)

        agent.add_episode(1)
        reward_window.push(episode_reward)
        length_window.push(episode_len)

    agent.export_saved_model()
    print("Done.")


if __name__ == "__main__":
    main()
