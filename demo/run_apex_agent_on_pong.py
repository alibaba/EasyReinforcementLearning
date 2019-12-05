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
import tensorflow as tf
import numpy as np

from easy_rl.agents import agents
from easy_rl.models import DQNModel
from easy_rl.utils.window_stat import WindowStat
from easy_rl.utils.gym_wrapper.atari_wrapper import make_atari, wrap_deepmind
from easy_rl.utils.vectorized_environment import VectorizedEnvironment

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("ps_hosts", "", "ps_hosts")
tf.flags.DEFINE_string("memory_hosts", "", "memory_hosts")
tf.flags.DEFINE_string("actor_hosts", "", "actor_hosts")
tf.flags.DEFINE_string("learner_hosts", "", "learn_hosts")
tf.flags.DEFINE_string("job_name", "", "job_name")
tf.flags.DEFINE_integer("task_index", -1, "task_index")
tf.flags.DEFINE_string("checkpoint_dir", "", "checkpoint_dir")
tf.flags.DEFINE_string("config", "", "path of the configuration")

# job partition
tf.flags.DEFINE_integer("num_actors", 0, "number of actors")
tf.flags.DEFINE_integer("num_memories", 0, "number of memories")


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
                    kernel_regularizer=tf.contrib.layers.l2_regularizer(
                        0.000001),
                    name="conv{}".format(i))
            flatten_hidden = tf.layers.flatten(inputs)
            hidden = tf.layers.dense(
                inputs=flatten_hidden,
                units=256,
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.000001),
                name="fc")
            logits = tf.layers.dense(
                hidden,
                units=6 * self.config.get("num_atoms", 1),
                activation=None)
            v_hidden = tf.layers.dense(
                flatten_hidden,
                units=256,
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.000001))
            v = tf.layers.dense(v_hidden, units=1, activation=None)
            return (logits, tf.squeeze(v))

    def preprocess_obs(self, obs, next_obs):
        obs = tf.cast(obs, tf.float32) / 1.0
        next_obs = tf.cast(next_obs, tf.float32) / 1.0
        return obs, next_obs


def main():
    with open(FLAGS.config, 'r') as ips:
        config = json.load(ips)

    job_name = FLAGS.job_name

    env = make_atari("PongNoFrameskip-v4")
    env = wrap_deepmind(
        env=env,
        frame_stack=True,
        clip_rewards=False,
        episode_life=True,
        wrap_frame=True,
        frame_resize=42)
    env.seed(0)

    # set different constant-eps for each actor
    num_actors = len(FLAGS.actor_hosts.split(','))
    eps = FLAGS.task_index * 1.0 / (num_actors - 1) * 0.7 + 0.02
    config["agent"]["constant_eps"] = eps
    print(config)

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
        custom_model=MyDQNModel,
        checkpoint_dir=None)
    all_cost = time.time()
    if job_name == "ps":
        print("ps starts===>")
        agent.join()
    elif job_name == "memory":
        start_tt = time.time()
        log_count = 0
        print("memory starts===>")
        while not agent.should_stop():
            agent.communicate()
            if time.time() - start_tt > log_count:
                log_count += 1
                print(agent._receive_count, "actor2mem_q:",
                      agent._actor2mem_q.qsize(), "mem2learner_q:",
                      agent._mem2learner_q.qsize())
                sys.stdout.flush()
    elif job_name == "actor":
        print("actor starts===>")
        start_tt = time.time()
        log_count = 0
        act_log_count = 0

        # create vectorized env
        def make_env(rank):
            def make_atari_env():
                env = make_atari("PongNoFrameskip-v4")
                env = wrap_deepmind(
                    env=env,
                    frame_stack=True,
                    clip_rewards=False,
                    episode_life=True,
                    wrap_frame=True,
                    frame_resize=42)
                env.seed(rank)
                return env

            return make_atari_env

        num_env = config["agent"].get("num_env", 1)
        vec_env = VectorizedEnvironment(
            make_env=make_env, num_env=num_env, seed=100 * FLAGS.task_index)

        act_cost = env_cost = send_cost = 0
        act_count = 0
        reward_window = WindowStat("reward", 10)
        length_window = WindowStat("length", 10)
        obs, actions, rewards, dones, next_obs = list(), list(), list(), list(
        ), list()
        agent.sync_vars()

        ob = vec_env.reset()
        episode_reward = np.zeros(num_env, )
        episode_len = np.zeros(num_env, )

        while not agent.should_stop():
            start = time.time()
            action, results = agent.act(ob, False)
            act_cost += time.time() - start
            act_count += 1

            start = time.time()
            next_ob, reward, done, info = vec_env.step(action)
            env_cost += time.time() - start

            obs.append(ob)
            actions.append(action)
            next_obs.append(next_ob)

            rewards.append(reward)
            dones.append(done)

            if agent.ready_to_send:
                start = time.time()
                agent.send_experience(
                    obs=obs,
                    actions=actions,
                    rewards=rewards,
                    next_obs=next_obs,
                    dones=dones,
                    is_vectorized_env=True,
                    num_env=num_env)
                send_cost += time.time() - start
                agent.sync_vars()

            ob = next_ob

            episode_reward += np.asarray(reward)
            episode_len += 1
            for i in range(num_env):
                if done[i]:
                    reward_window.push(episode_reward[i])
                    length_window.push(episode_len[i])
                    episode_reward[i] = .0
                    episode_len[i] = 0
            total_cost = time.time() - start_tt
            if int(total_cost / 5) > log_count:
                log_count += 1
                print("act_count:", act_count, "actor2mem_q:",
                      agent._actor2mem_q.qsize(), "total_cost:", total_cost,
                      reward_window)
                print(length_window)
            if int((act_count * num_env) / 10000) > act_log_count:
                act_log_count += 1
                print("timestep:", act_log_count * 10000, reward_window)
        print(
            "actor has terminated...now we need to close the vectorized environment"
        )
        sys.stdout.flush()
        vec_env.close()

    elif job_name == "learner":
        print("learner starts===>")
        start_tt = time.time()
        rece_cost = learn_cost = 0
        train_count = 0
        while not agent.should_stop():
            start = time.time()
            batch_data = agent.receive_experience()
            rece_cost += time.time() - start
            if batch_data:
                start = time.time()
                extra_data = agent.learn(batch_data)
                learn_cost += time.time() - start
                train_count += 1
                print("learning {}".format(extra_data), "receive_q:",
                      agent._receive_q.qsize())
                print("train_count:", train_count, "total:",
                      time.time() - start_tt)
                sys.stdout.flush()

    else:
        raise ValueError("Invalid job_name.")
    all_cost = time.time() - all_cost
    print("done. all_cost:", all_cost)


if __name__ == "__main__":
    main()
