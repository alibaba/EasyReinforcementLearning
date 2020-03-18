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
import time
import numpy as np
from gym.spaces import Box, Discrete

from easy_rl.agents import agents
from easy_rl.models import BatchDQNModel
from easy_rl.utils.window_stat import WindowStat
from easy_rl.utils.inverse_propensity_score import ips_eval

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("train_data_file", "", "tables names")
tf.flags.DEFINE_string("eval_data_file", "", "tables names")

MODEL_CONFIG = dict(
    # specific
    type="BCQ",
    n_step=3,
    dueling=False,
    double_q=True,

    ratio_threshold=0.3,

    # common
    gamma=0.95,
    init_lr=1e-3,
    lr_strategy_spec={
        'type': 'exponential_decay',
        'decay_steps': 500,
        'decay_rate': 0.8
    },
    clone_init_lr=1e-2,
    clone_lr_strategy_spec={
        'type': 'exponential_decay',
        'decay_steps': 1000,
        'decay_rate': 0.9
    },
    global_norm_clip=0.01)

AGENT_CONFIG = dict(
    type="BA",
    buffer_size=50000,
    learning_starts=1000,
    batch_size=256,
    sync_target_frequency=600,
)
np.random.seed(0)


class MyBCQModel(BatchDQNModel):
    def _generative_model(self, input_obs, scope="generative_model"):
        with tf.variable_scope(name_or_scope=scope):

            h1 = tf.layers.dense(
                input_obs,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=0),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
            h2 = tf.layers.dense(
                h1,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=2),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
            logits = tf.layers.dense(
                h2,
                units=2,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=5))

            return logits

    def _encode_obs(self, input_obs, scope="encode_obs"):
        with tf.variable_scope(name_or_scope=scope):
            h1 = tf.layers.dense(
                input_obs,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=0),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
            h2 = tf.layers.dense(
                h1,
                units=64,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=0),
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0001))
            logits = tf.layers.dense(
                h2,
                units=2,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=0))
            return logits

class offline_env(object):

    def __init__(self, file_name, batch_size=128, n_step=1):
        # create an offline_env to do fake interaction with agent
        self.num_epoch = 0
        self.num_record = 0
        self._offset = 0

        # how many records to read from table at one time
        self.batch_size = batch_size
        # number of step to reserved for n-step dqn
        self.n_step = n_step

        # defined the shape of observation and action
        # we follow the definition of gym.spaces
        # `Box` for continue-space, `Discrete` for discrete-space and `Dict` for multiple input
        # actually low/high limitation will not be used by agent but required by gym.spaces
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(4,))
        self.action_space = Discrete(n=2)

        fr = open(file_name)
        self.data = fr.readlines()
        self.num_record = len(self.data)
        fr.close()

    def parse_tuple_data(self, lines):

        obs = []
        actions = []
        rewards = []
        dones = []
        next_obs = []
        for line in lines:
            state, action, reward, terminal, next_state = line.strip().split('\t')
            obs.append([float(e) for e in state.split(',')])
            actions.append(int(action))
            rewards.append(float(reward))
            dones.append(int(terminal))
            next_obs.append([float(e) for e in next_state.split(',')])

        dict_data = dict(
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_obs=next_obs)

        return dict_data

    def reset(self):
        read_batch_size = self.batch_size + self.n_step - 1

        if self._offset + read_batch_size > self.num_record:
            self._offset = 0

        dict_data = self.parse_tuple_data(self.data[self._offset:self._offset + read_batch_size])
        self._offset += read_batch_size

        return dict_data

def main():
    # create offline_env
    env = offline_env(FLAGS.train_data_file, batch_size=128, n_step=MODEL_CONFIG.get("n_step", 1))
    eval_env = offline_env(FLAGS.eval_data_file, batch_size=128, n_step=MODEL_CONFIG.get("n_step", 1))

    agent_class = agents[AGENT_CONFIG["type"]]
    agent = agent_class(
        env.observation_space,
        env.action_space,
        AGENT_CONFIG,
        MODEL_CONFIG,
        distributed_spec={},
        export_dir="bcq_tmp",
        checkpoint_dir="bcq_tmp",
        custom_model=MyBCQModel)

    clone_loss_window = WindowStat("clone_loss", 50)
    clone_reg_loss_window = WindowStat("clone_reg_loss", 50)
    loss_window = WindowStat("loss", 50)

    total_cost = time.time()
    clone_learn_count = 0

    # first, train a generative model by behavior clone
    for i in range(1000):
        table_data = env.reset()
        # store raw data in replay buffer
        agent.send_experience(
            obs=table_data["obs"],
            actions=table_data["actions"],
            rewards=table_data["rewards"],
            dones=table_data["dones"],
            next_obs=table_data["next_obs"])

        # sample from replay buffer
        # the size of sampled data is equal to `AGENT_CONFIG["batch_size"]`
        batch_data = agent.receive_experience()
        clone_loss, clone_reg_loss = agent.behavior_learn(batch_data=batch_data)
        clone_learn_count += 1
        clone_loss_window.push(clone_loss)
        clone_reg_loss_window.push(clone_reg_loss)
        if i % 50 == 0:
            print(clone_loss_window)
            print(clone_reg_loss_window)

    # eval generative model
    all_clone_act, gd_act = [], []
    for i in range(100):
        table_data = env.reset()
        clone_act = agent.behavior_act(table_data["obs"])
        all_clone_act.extend(np.argsort(-1.0 * clone_act, axis=1).tolist())
        gd_act.extend(table_data["actions"])
    acc1 = np.sum(np.array(all_clone_act)[:, 0] == np.array(gd_act))*1.0/len(gd_act)
    print("acc @top1:", acc1)

    # second, train bcq
    agent.reset_global_step()

    epochs_to_end = 10
    max_globel_steps_to_end = 10000
    learn_count = 0
    env.num_epoch = 0
    while env.num_epoch < epochs_to_end and learn_count < max_globel_steps_to_end:
        table_data = env.reset()

        # store raw data in replay buffer
        agent.send_experience(
            obs=table_data["obs"],
            actions=table_data["actions"],
            rewards=table_data["rewards"],
            dones=table_data["dones"],
            next_obs=table_data["next_obs"])

        # sample from replay buffer
        # the size of sampled data is equal to `AGENT_CONFIG["batch_size"]`
        batch_data = agent.receive_experience()
        # update the model
        res = agent.learn(batch_data)
        # record the loss
        loss_window.push(res["loss"])
        learn_count += 1

        if AGENT_CONFIG.get("prioritized_replay", False):
            # update priorities
            agent.update_priorities(
                indexes=batch_data["indexes"],
                td_error=res["td_error"])

        if learn_count % 50 == 0:
            print("learn_count:", learn_count)
            print(loss_window)

            # offline evaluation
            batch_weights = []
            batch_rewards = []
            eval_num = 50
            for _ in range(eval_num):
                batch_data = eval_env.reset()

                importance_ratio = agent.importance_ratio(batch_data)
                batch_weights.append(importance_ratio)
                batch_rewards.append(batch_data["rewards"])
            ips, ips_sw, wips, wips_sw, wips_sw_mean = ips_eval(
                batch_weights=batch_weights, batch_rewards=batch_rewards, gamma=MODEL_CONFIG.get("gamma", 0.95))

            agent.add_extra_summary({agent.model.ips_score_op:ips,
                                     agent.model.ips_score_stepwise_op:ips_sw,
                                     agent.model.wnorm_ips_score_op:wips,
                                     agent.model.wnorm_ips_score_stepwise_op:wips_sw,
                                     agent.model.wnorm_ips_score_stepwise_mean_op:wips_sw_mean})
            print("[IPS Policy Evaluation @learn_count={}] ips={}, ips_stepwise={}, wnorm_ips={}, wnorm_ips_stepwise={}, wnorm_ips_stepwise_mean={}".format(
                learn_count, ips, ips_sw, wips, wips_sw, wips_sw_mean))

        if learn_count % 2000 == 0:
            # export saved model at any time
            # AssertionError will occur if the export_dir already exists.
            agent.export_saved_model("bcq_export_dir{}".format(learn_count))

        if learn_count % 200 == 0:
            # test with simulator
            gym_env = gym.make("CartPole-v0")
            for ix in range(10):
                ob = gym_env.reset()
                done = False
                episode_reward = .0

                while not done:
                    action, results = agent.act(
                        [ob], deterministic=False, use_perturbed_action=False)

                    next_ob, reward, done, info = gym_env.step(action[0])
                    episode_reward += reward
                    ob = next_ob
                print("train@", learn_count, "test@", ix, "reward:", episode_reward)

    print("Done.", "num_epoch:", env.num_epoch, "learn_count:", learn_count,
          "total_cost:", time.time() - total_cost)


if __name__ == "__main__":
    main()
