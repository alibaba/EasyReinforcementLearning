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
import gym, json, time, sys

from easy_rl.agents import agents
from easy_rl.models import PGModel
from easy_rl.utils.window_stat import WindowStat

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("ps_hosts", "", "ps_hosts")
tf.flags.DEFINE_string("worker_hosts", "", "learn_hosts")
tf.flags.DEFINE_string("job_name", "", "job_name")
tf.flags.DEFINE_integer("task_index", -1, "task_index")
tf.flags.DEFINE_string("checkpoint_dir", "", "checkpoint_dir")
tf.flags.DEFINE_string("config", "", "path of the configuration")


class MyPGModel(PGModel):
    def _encode_obs(self, input_obs, scope="encode_obs"):
        with tf.variable_scope(name_or_scope=scope):
            h1 = tf.layers.dense(input_obs, units=64, activation=tf.nn.relu)
            h2 = tf.layers.dense(h1, units=64, activation=tf.nn.relu)
            logits = tf.layers.dense(h2, units=2, activation=None)

            v1 = tf.layers.dense(
                input_obs,
                units=256,
                activation=tf.nn.relu,
                name="logits_to_value")
            v2 = tf.layers.dense(
                v1, units=64, activation=tf.nn.relu, name="value_hidden")
            v3 = tf.layers.dense(v2, units=1, activation=None, name="value")
            v = tf.squeeze(v3, [-1])

            return (logits, v)


np.random.seed(0)


def main():
    with open(FLAGS.config, 'r') as ips:
        config = json.load(ips)
        print(config)

    env = gym.make("CartPole-v0")
    env.seed(0)

    agent_class = agents[config["agent"]["type"]]
    agent = agent_class(
        env.observation_space,
        env.action_space,
        agent_config=config["agent"],
        model_config=config["model"],
        distributed_spec={
            "ps_hosts": FLAGS.ps_hosts,
            "worker_hosts": FLAGS.worker_hosts,
            "job_name": FLAGS.job_name,
            "task_index": FLAGS.task_index
        },
        custom_model=MyPGModel,
        checkpoint_dir="")

    all_cost = time.time()
    if FLAGS.job_name == "ps":
        print("ps starts===>")
        agent.join()
    elif FLAGS.job_name == "worker":
        print("actor starts===>")
        act_count = 0
        train_count = 0
        reward_window = WindowStat("reward", 50)
        length_window = WindowStat("length", 50)
        obs, actions, rewards, dones, value_preds = list(), list(), list(
        ), list(), list()

        while not agent.should_stop():
            ob = env.reset()
            done = False
            episode_reward = 0.0
            episode_len = 0.0

            while not done and not agent.should_stop():
                action, results = agent.act([ob], False)

                act_count += 1

                new_ob, reward, done, info = env.step(action[0])

                obs.append(ob)
                actions.append(action[0])
                rewards.append(0.1 * reward)
                dones.append(done)
                value_preds.append(results["value_preds"][0])

                if agent.ready_to_send:
                    agent.send_experience(
                        obs=obs,
                        actions=actions,
                        rewards=rewards,
                        dones=dones,
                        value_preds=value_preds)
                    batch_data = agent.receive_experience()
                    if batch_data:
                        extra_data = agent.learn(batch_data)
                        train_count += 1
                        print("learning {}".format(extra_data))

                ob = new_ob
                episode_reward += reward
                episode_len += 1
            print("act_count:", act_count)
            reward_window.push(episode_reward)
            length_window.push(episode_len)
            print(reward_window)
            print(length_window)
            sys.stdout.flush()

        if FLAGS.task_index == 0:
            agent.export_saved_model(export_dir="a3c_export_dir")
            print("export savedmodel finish")
    else:
        raise ValueError("Invalid job_name.")
    all_cost = time.time() - all_cost
    print("done. all_cost:", all_cost)


if __name__ == "__main__":
    main()
