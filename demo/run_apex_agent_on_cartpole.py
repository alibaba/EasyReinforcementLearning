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
from __future__ import division
from __future__ import print_function

import sys, time, json
import gym
import numpy as np
import tensorflow as tf

from easy_rl.agents import agents
from easy_rl.models import DQNModel
from easy_rl.utils.window_stat import WindowStat

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("ps_hosts", "", "ps_hosts")
tf.flags.DEFINE_string("memory_hosts", "", "memory_hosts")
tf.flags.DEFINE_string("actor_hosts", "", "actor_hosts")
tf.flags.DEFINE_string("learner_hosts", "", "learn_hosts")
tf.flags.DEFINE_string("job_name", "", "job_name")
tf.flags.DEFINE_integer("task_index", -1, "task_index")
tf.flags.DEFINE_string("checkpoint_dir", "", "checkpoint_dir")
tf.flags.DEFINE_string("config", "", "path of the configuration")

# set num_atoms > 1 if use Distributional-Dqn
num_atoms = 11


class MyDQN(DQNModel):
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
                units=2 * num_atoms,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=0.01, seed=0))
            return logits


np.random.seed(0)


def main(_):
    with open(FLAGS.config, 'r') as ips:
        config = json.load(ips)
        print(config)

    env = gym.make("CartPole-v0")
    env.seed(0)

    agent_class = agents[config["agent"]["type"]]
    agent = agent_class(
        env.observation_space,
        env.action_space,
        config["agent"],
        config["model"],
        distributed_spec={
            "ps_hosts": FLAGS.ps_hosts,
            "memory_hosts": FLAGS.memory_hosts,
            "actor_hosts": FLAGS.actor_hosts,
            "learner_hosts": FLAGS.learner_hosts,
            "job_name": FLAGS.job_name,
            "task_index": FLAGS.task_index
        },
        custom_model=MyDQN)

    if FLAGS.job_name == "ps":
        print("ps starts===>")
        agent.join()
    elif FLAGS.job_name == "memory":
        print("memory starts===>")
        while not agent.should_stop():
            agent.communicate()
            sys.stdout.flush()
    elif FLAGS.job_name == "actor":
        print("actor starts===>")
        reward_window = WindowStat("reward", 50)
        length_window = WindowStat("length", 50)
        obs, actions, rewards, new_obs, dones = list(), list(), list(), list(
        ), list()
        agent.sync_vars()

        while not agent.should_stop():
            ob = env.reset()
            done = False
            episode_reward = .0
            episode_len = 0

            while not done and not agent.should_stop():
                action, results = agent.act([ob], False)

                new_ob, reward, done, info = env.step(action[0])

                obs.append(ob)
                actions.append(action[0])
                rewards.append(reward)
                new_obs.append(new_ob)
                dones.append(done)
                if agent.ready_to_send:
                    agent.send_experience(
                        obs=obs,
                        actions=actions,
                        rewards=rewards,
                        next_obs=new_obs,
                        dones=dones)
                    agent.sync_vars()
                ob = new_ob
                episode_reward += reward
                episode_len += 1

            reward_window.push(episode_reward)
            length_window.push(episode_len)
            print(reward_window)
            print(length_window)
            sys.stdout.flush()
    elif FLAGS.job_name == "learner":
        print("learner starts===>")
        while not agent.should_stop():
            batch_data = agent.receive_experience()
            if batch_data:
                extra_data = agent.learn(batch_data)
                print("learning {}".format(extra_data))
                sys.stdout.flush()
    else:
        raise ValueError("Invalid job_name.")

    print("done.")


if __name__ == "__main__":
    tf.app.run()
