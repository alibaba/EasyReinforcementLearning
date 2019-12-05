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
import numpy as np
import gym
import tensorflow as tf

from easy_rl.agents import agents
from easy_rl.models import EvolutionStrategy
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


def rollout(agent, env, episode_num=20, use_perturbed_action=False):
    rewards = []
    episode_lens = []
    for i in range(episode_num):
        ob = env.reset()
        done = False
        episode_reward = .0
        episode_len = 0

        while not done and not agent.should_stop():
            action, results = agent.act(
                [ob], True, use_perturbed_action=use_perturbed_action)
            new_ob, reward, done, info = env.step(action[0])

            episode_reward += reward
            ob = new_ob
            episode_len += 1

        rewards.append(episode_reward)
        episode_lens.append(episode_len)

    return rewards, episode_lens


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
        custom_model=MyESmodel,
        checkpoint_dir=None)
    all_cost = time.time()
    if FLAGS.job_name == "ps":
        print("ps starts===>")
        agent.join()
    elif FLAGS.job_name == "memory":
        print("memory starts===>")
        while not agent.should_stop():
            agent.communicate()
            print("communicating")
            time.sleep(0.1)
    elif FLAGS.job_name == "actor":
        print("actor starts===>")
        reward_window = WindowStat("reward", 50)
        length_window = WindowStat("length", 50)

        perturbation_scale = 0.1
        run_episode_per_perturbation = config["agent"].get(
            "run_episode_per_perturbation", 1)

        seeds, rewards, perturbation_scales = list(), list(), list()
        is_positive = False
        returns = list()

        agent.sync_vars()

        episode_count = 0
        try:
            while not agent.should_stop():

                # do perturbation
                is_positive = False if is_positive else True

                # each seed will be used twice
                seed = np.random.randint(1000000) if is_positive else seeds[-1]
                perturbation_scale = max(
                    perturbation_scale * (1 - episode_count / 2000.0), 0.02)

                feed = agent.behavior_model.perturbation_feed
                fetch = [agent.behavior_model.reset_perturbation_op]

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

                rewards, episode_lens = rollout(
                    agent,
                    env,
                    episode_num=run_episode_per_perturbation,
                    use_perturbed_action=True)
                episode_count += run_episode_per_perturbation

                # calculate the average reward from a specific perturbation with one direction
                if len(returns) == 0:
                    returns.append([np.mean(rewards)])
                elif len(returns[-1]) < 2:
                    returns[-1].append(np.mean(rewards))
                else:
                    returns.append([np.mean(rewards)])

                if len(returns) == agent.config.get(
                        'sample_batch_size', 100) and len(returns[-1]) == 2:
                    # send out the results for the latest `sample_batch_size` * 2 trials
                    print(reward_window)
                    assert len(seeds) == len(returns)
                    assert len(perturbation_scales) == len(returns)

                    agent.send_experience(**dict(
                        perturbation_seeds=seeds,
                        perturbation_scales=perturbation_scales,
                        returns=returns))

                    # reset the direction
                    is_positive = False

                    # synchronize the weights from parameter server to local behavior_model
                    agent.sync_vars()

                    # do evaluation for 20 episode
                    evaluation_num = 20
                    evl_returns, _ = rollout(
                        agent,
                        env,
                        episode_num=evaluation_num,
                        use_perturbed_action=False)
                    print(
                        "evaluation at episode:", episode_count,
                        ",avg episode reward of {} evaluation:".format(
                            evaluation_num), np.mean(evl_returns))

                reward_window.push(rewards)
                length_window.push(episode_lens)
                if episode_count % 50 == 0:
                    print(reward_window)
                    print(length_window)
                    sys.stdout.flush()
        except tf.errors.OutOfRangeError as e:
            print("memory has stopped.")

    elif FLAGS.job_name == "learner":
        print("learner starts===>")
        train_count = 0
        try:
            while not agent.should_stop():
                batch_data = agent.receive_experience()
                if batch_data:
                    extra_data = agent.learn(batch_data)
                    train_count += 1
                    print("learning {}".format(extra_data))
                    sys.stdout.flush()
        except tf.errors.OutOfRangeError as e:
            print("memory has stopped.")
    else:
        raise ValueError("Invalid job_name.")
    all_cost = time.time() - all_cost
    print("done. all_cost:", all_cost)


if __name__ == "__main__":
    main()
