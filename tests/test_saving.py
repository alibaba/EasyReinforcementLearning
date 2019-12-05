from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import unittest
import numpy as np
import tensorflow as tf
import gym

from easy_rl.agents import agents
from easy_rl.models import DQNModel
from easy_rl.utils.window_stat import WindowStat

DQN_MODEL_CONFIG = dict(
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

DQN_AGENT_CONFIG = dict(
    type="Agent",
    sample_batch_size=4,
    buffer_size=50000,
    learning_starts=500,
    prioritized_replay=False,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta=0.4,
    batch_size=256,
    sync_target_frequency=100,
    exploration_timesteps=40000,
    perturbation_frequency=40,  # recommend to set to 50
    noise_kl_episodes=300  # after 300 episodes kl_threshold will decay to 1e-4
)


class SavingTest(unittest.TestCase):
    """Save checkpoint and export saved model.
    """

    def doTestCkpt(self):
        trial_timestamp = time.strftime("%Y%m%d-%H%M%S")
        np.random.seed(0)
        env = gym.make("CartPole-v0")
        env.seed(0)
        dqn_g = tf.Graph()
        with dqn_g.as_default():
            tf.set_random_seed(123)
            agent = agents[DQN_AGENT_CONFIG["type"]](
                env.observation_space,
                env.action_space,
                DQN_AGENT_CONFIG,
                DQN_MODEL_CONFIG,
                checkpoint_dir="ckpt_dir_{}".format(trial_timestamp),
                distributed_spec={})
        reward_window = WindowStat("reward", 50)
        obs, actions, rewards, next_obs, dones = list(), list(), list(), list(
        ), list()
        act_count = 0

        for i in range(500):
            ob = env.reset()
            done = False
            episode_reward = .0

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

                    if DQN_AGENT_CONFIG.get("prioritized_replay", False):
                        agent.update_priorities(
                            indexes=batch_data["indexes"],
                            td_error=res["td_error"])

                ob = next_ob
                episode_reward += reward
                if act_count % 1024 == 0:
                    print("timestep:", act_count, reward_window)

            agent.add_episode(1)
            reward_window.push(episode_reward)

        prev_perf = reward_window.stats()["reward_mean"]
        print("Performance before saving is {}".format(prev_perf))

        new_dqn_g = tf.Graph()
        with new_dqn_g.as_default():
            agent = agents[DQN_AGENT_CONFIG["type"]](
                env.observation_space,
                env.action_space,
                DQN_AGENT_CONFIG,
                DQN_MODEL_CONFIG,
                checkpoint_dir="ckpt_dir_{}".format(trial_timestamp),
                distributed_spec={})
        reward_window = WindowStat("reward", 10)
        ob = env.reset()
        for i in range(10):
            ob = env.reset()
            done = False
            episode_reward = .0

            while not done:
                action, results = agent.act(
                    [ob], deterministic=True, use_perturbed_action=False)

                next_ob, reward, done, info = env.step(action[0])
                act_count += 1

                ob = next_ob
                episode_reward += reward

            agent.add_episode(1)
            reward_window.push(episode_reward)

        cur_perf = reward_window.stats()["reward_mean"]
        print("Performance after restore is {}".format(cur_perf))
        return prev_perf - cur_perf

    def testCkpt(self):
        mean_episode_reward_diff = self.doTestCkpt()
        self.assertTrue(mean_episode_reward_diff < 10)

    def doTestSavedModel(self):
        trial_timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_dir = "model_dir_{}".format(trial_timestamp)
        os.system("mkdir {}".format(model_dir))

        np.random.seed(0)
        env = gym.make("CartPole-v0")
        env.seed(0)
        dqn_g = tf.Graph()
        with dqn_g.as_default():
            tf.set_random_seed(123)
            agent = agents[DQN_AGENT_CONFIG["type"]](
                env.observation_space,
                env.action_space,
                DQN_AGENT_CONFIG,
                DQN_MODEL_CONFIG,
                export_dir=model_dir,
                distributed_spec={})
        reward_window = WindowStat("reward", 50)
        obs, actions, rewards, next_obs, dones = list(), list(), list(), list(
        ), list()
        act_count = 0

        for i in range(500):
            ob = env.reset()
            done = False
            episode_reward = .0

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

                    if DQN_AGENT_CONFIG.get("prioritized_replay", False):
                        agent.update_priorities(
                            indexes=batch_data["indexes"],
                            td_error=res["td_error"])

                ob = next_ob
                episode_reward += reward
                if act_count % 1024 == 0:
                    print("timestep:", act_count, reward_window)

            agent.add_episode(1)
            reward_window.push(episode_reward)

        prev_perf = reward_window.stats()["reward_mean"]
        print("Performance before saving is {}".format(prev_perf))

        with tf.Session() as sess:
            path = model_dir
            MetaGraphDef = tf.saved_model.loader.load(
                sess, tags=[sm.tag_constants.SERVING], export_dir=path)

            # get SignatureDef protobuf
            SignatureDef_d = MetaGraphDef.signature_def
            SignatureDef = SignatureDef_d["predict_results"]

            # get inputs/outputs TensorInfo protobuf
            ph_inputs = {}
            for name, ts_info in SignatureDef.inputs.items():
                ph_inputs[name] = sm.utils.get_tensor_from_tensor_info(
                    ts_info, sess.graph)

            outputs = {}
            for name, ts_info in SignatureDef.outputs.items():
                outputs[name] = sm.utils.get_tensor_from_tensor_info(
                    ts_info, sess.graph)

            for name, ph in ph_inputs.items():
                print(name, ph)

            for name, ts in outputs.items():
                print(name, ts)

            reward_window = WindowStat("reward", 10)
            for i in range(10):
                ob = env.reset()
                done = False
                episode_reward = .0

                while not done:
                    action = sess.run(
                        outputs["output_actions"],
                        feed_dict={
                            ph_inputs["obs_ph"]: [np.asarray(ob)],
                            ph_inputs["deterministic_ph"]: True
                        })
                    next_ob, reward, done, info = env.step(action[0])
                    episode_reward += reward
                    ob = next_ob

                reward_window.push(episode_reward)

        cur_perf = reward_window.stats()["reward_mean"]
        print("Performance after restore is {}".format(cur_perf))
        return prev_perf - cur_perf

    def testSavedModel(self):
        mean_episode_reward_diff = self.doTestCkpt()
        self.assertTrue(mean_episode_reward_diff < 10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
