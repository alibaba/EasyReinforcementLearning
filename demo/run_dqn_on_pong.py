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

from easy_rl.agents import agents
from easy_rl.models import DQNModel
from easy_rl.utils.window_stat import WindowStat
from easy_rl.utils.gym_wrapper.atari_wrapper import make_atari, wrap_deepmind

MODEL_CONFIG = dict(
    # specific
    type="DQN",
    n_step=3,
    dueling=True,
    double_q=True,
    num_atoms=1,  # recommend to set 11 to run distributional dqn
    v_min=0,
    v_max=25,

    # common
    parameter_noise=False,  # set True to use parameter_noise
    gamma=0.99,
    init_lr=5e-4,
    lr_strategy_spec={
        'type': 'exponential_decay',
        'decay_steps': 100000,
        'decay_rate': 0.9
    },
    global_norm_clip=0.6)

AGENT_CONFIG = dict(
    type="Agent",
    sample_batch_size=4,
    buffer_size=20000,
    learning_starts=10000,
    prioritized_replay=False,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta=0.4,
    batch_size=32,
    sync_target_frequency=500,
    exporation_timesteps=600000,
    perturbation_frequency=40,  # recommend to set to 50
    noise_kl_episodes=300  # after 300 episodes kl_threshold will decay to 1e-4
)
np.random.seed(1)


class MyDQNModel(DQNModel):
    def _encode_obs(self, input_obs, scope="encode_obs"):
        _conv_filters = ((16, (4, 4), 2, 'same'), (32, (4, 4), 2, 'same'),
                         (256, (11, 11), 1, 'valid'))
        inputs = input_obs
        with tf.variable_scope(name_or_scope=scope):
            for i, (filters_, kernel_size, stride, padding) in enumerate(
                    _conv_filters, 1):
                inputs = tf.layers.conv2d(
                    inputs=inputs,
                    filters=filters_,
                    kernel_size=kernel_size,
                    strides=stride,
                    padding=padding,
                    activation=tf.nn.relu,
                    name="conv{}".format(i))
            flatten_hidden = tf.layers.flatten(inputs)
            hidden = tf.layers.dense(
                inputs=flatten_hidden,
                units=256,
                activation=tf.nn.relu,
                name="fc")
            logits = tf.layers.dense(
                hidden, units=6 * MODEL_CONFIG["num_atoms"], activation=None)
            v_hidden = tf.layers.dense(
                flatten_hidden, units=256, activation=tf.nn.relu)
            v = tf.layers.dense(v_hidden, units=1, activation=None)
            return (logits, tf.squeeze(v))

    def preprocess_obs(self, obs, next_obs):
        obs = tf.cast(obs, tf.float32) / 255.0
        next_obs = tf.cast(next_obs, tf.float32) / 255.0
        return obs, next_obs


def main():
    env = make_atari("PongNoFrameskip-v4")
    env = wrap_deepmind(
        env=env,
        frame_stack=True,
        clip_rewards=False,
        episode_life=True,
        wrap_frame=True,
        frame_resize=42)

    agent_class = agents[AGENT_CONFIG["type"]]
    agent = agent_class(
        env.observation_space,
        env.action_space,
        AGENT_CONFIG,
        MODEL_CONFIG,
        distributed_spec={},
        custom_model=MyDQNModel)

    reward_window = WindowStat("reward", 10)
    length_window = WindowStat("length", 10)
    loss_window = WindowStat("loss", 10)
    obs, actions, rewards, next_obs, dones = list(), list(), list(), list(
    ), list()

    for i in range(2000):
        ob = env.reset()
        ob = np.asarray(ob)
        done = False
        episode_reward = .0
        episode_len = 0

        while not done:
            action, results = agent.act(
                [ob], deterministic=False, use_perturbed_action=False)

            next_ob, reward, done, info = env.step(action[0])
            next_ob = np.asarray(next_ob)

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
        agent.add_episode(1)
        reward_window.push(episode_reward)
        length_window.push(episode_len)
        if i % 10 == 0:
            print('episode at', i)
            print(reward_window)
            print(length_window)
            print(loss_window)

    print("Done.")


if __name__ == "__main__":
    main()
