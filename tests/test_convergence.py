from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf
import gym

from easy_rl.agents import agents
from easy_rl.models import DQNModel
from easy_rl.utils.window_stat import WindowStat
from easy_rl.models import EvolutionStrategy

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

DDPG_MODEL_CONFIG = dict(
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

DDPG_AGENT_CONFIG = dict(
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

PPO_MODEL_CONFIG = dict(
    # specific
    type="PPO",

    # common
    init_lr=1e-3,
    lr_strategy_spec={
        'type': 'exponential_decay',
        'decay_steps': 100,
        'decay_rate': 0.9
    },
    global_norm_clip=40)

PPO_AGENT_CONFIG = dict(
    type="Agent",
    sample_batch_size=64,
    batch_size=128,
    sub_train_batch=64,
    train_epochs=2,

    # gae
    gamma=0.9,
    lambda_=0.5,
    use_gae=True,
)

ES_MODEL_CONFIG = dict(
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

ES_AGENT_CONFIG = dict(
    type="Agent",
    sample_batch_size=100,
    batch_size=100,
)


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


class ConvergenceTest(unittest.TestCase):
    """Run commonly used algorithms in single process mode.
    Validate their convergence on classic simulators.
    """

    def doTestDQN(self):
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
                distributed_spec={})
        reward_window = WindowStat("reward", 25)
        obs, actions, rewards, next_obs, dones = list(), list(), list(), list(
        ), list()
        act_count = 0

        for i in range(600):
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
        return reward_window.stats()["reward_mean"]

    def testDQN(self):
        mean_episode_reward = self.doTestDQN()
        self.assertTrue(mean_episode_reward >= 190)

    def doTestDDPG(self):
        np.random.seed(0)
        env = gym.make("Pendulum-v0")
        env.seed(0)
        ddpg_g = tf.Graph()
        with ddpg_g.as_default():
            tf.set_random_seed(123)
            agent = agents[DDPG_AGENT_CONFIG["type"]](
                env.observation_space,
                env.action_space,
                DDPG_AGENT_CONFIG,
                DDPG_MODEL_CONFIG,
                distributed_spec={})
        reward_window = WindowStat("reward", 25)
        obs, actions, rewards, next_obs, dones = list(), list(), list(), list(
        ), list()
        act_count = 0

        for i in range(200):
            ob = env.reset()
            done = False
            episode_reward = .0

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

                    if DDPG_AGENT_CONFIG.get("prioritized_replay", False):
                        agent.update_priorities(
                            indexes=batch_data["indexes"],
                            td_error=res["td_error"])

                ob = next_ob
                episode_reward += reward
                if act_count % 1024 == 0:
                    print("timestep:", act_count, reward_window)

            agent.add_episode(1)
            reward_window.push(episode_reward)

        return reward_window.stats()["reward_mean"]

    def testDDPG(self):
        mean_episode_reward = self.doTestDDPG()
        self.assertTrue(mean_episode_reward >= -300)

    def doTestPPO(self):
        env = gym.make("CartPole-v0")
        env.seed(0)
        ppo_g = tf.Graph()
        with ppo_g.as_default():
            tf.set_random_seed(123)
            agent = agents[PPO_AGENT_CONFIG["type"]](
                env.observation_space,
                env.action_space,
                PPO_AGENT_CONFIG,
                PPO_MODEL_CONFIG,
                distributed_spec={})

        reward_window = WindowStat("reward", 25)
        obs, actions, rewards, next_obs, dones, value_preds, logits = list(
        ), list(), list(), list(), list(), list(), list()
        act_count = 0

        for i in range(300):
            ob = env.reset()
            done = False
            episode_reward = .0

            while not done:
                action, results = agent.act([ob], False)
                next_ob, reward, done, info = env.step(action[0])
                act_count += 1

                obs.append(ob)
                actions.append(action[0])
                rewards.append(0.1 * reward)
                next_obs.append(next_ob)
                dones.append(done)

                logits.append(results["logits"][0])
                value_preds.append(results["value_preds"][0])
                if agent.ready_to_send:
                    agent.send_experience(
                        obs=obs,
                        actions=actions,
                        rewards=rewards,
                        dones=dones,
                        next_obs=next_obs,
                        value_preds=value_preds,
                        logits=logits)
                if agent.ready_to_receive:
                    batch_data = agent.receive_experience()
                    res = agent.learn(batch_data)

                ob = next_ob
                episode_reward += reward
                if act_count % 1024 == 0:
                    print("timestep:", act_count, reward_window)

            reward_window.push(episode_reward)

        return reward_window.stats()["reward_mean"]

    def testPPO(self):
        mean_episode_reward = self.doTestPPO()
        self.assertTrue(mean_episode_reward >= 190)

    def doTestES(self):
        np.random.seed(0)
        env = gym.make("CartPole-v0")
        env.seed(0)
        es_g = tf.Graph()
        with es_g.as_default():
            tf.set_random_seed(123)
            agent = agents[ES_AGENT_CONFIG["type"]](
                env.observation_space,
                env.action_space,
                ES_AGENT_CONFIG,
                ES_MODEL_CONFIG,
                distributed_spec={},
                custom_model=MyESmodel)
        reward_window = WindowStat("reward", 25)

        perturbation_scale = 0.1

        seeds, rewards, perturbation_scales = list(), list(), list()
        is_positive_direction = list()

        episode_per_perturbation = 1

        returns = list()

        for i in range(5000):
            ob = env.reset()
            done = False
            episode_reward = .0

            if i % episode_per_perturbation == 0:
                # perturb parameters every `episode_per_seed` episodes
                is_positive = True if len(
                    is_positive_direction
                ) == 0 else is_positive_direction[-1] != True

                # each seed twice
                seed = np.random.randint(1000000) if is_positive else seeds[-1]
                perturbation_scale = max(perturbation_scale * (1 - i / 2000.0),
                                         0.02)

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
                action, result = agent.act(
                    [ob], True, use_perturbed_action=True)

                next_ob, reward, done, info = env.step(action[0])

                ob = next_ob
                episode_reward += reward

            rewards.append(episode_reward)
            reward_window.push(episode_reward)
            if len(rewards) == episode_per_perturbation:
                returns.append(np.mean(rewards))
                rewards = []
                if len(returns) == 2 * agent.config.get(
                        'sample_batch_size', 100):
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
                    for j in range(10):
                        done = False
                        ob = env.reset()
                        episode_reward = 0
                        while not done:
                            action, result = agent.act(
                                [ob], True, use_perturbed_action=False)

                            next_ob, reward, done, info = env.step(action[0])

                            ob = next_ob
                            episode_reward += reward
                        test_rewards.append(episode_reward)
                    print("[evaluation] average reward of 20 episodes:",
                          np.mean(test_rewards))
                    print('train at ', i)

        return np.mean(test_rewards)

    def testES(self):
        mean_episode_reward = self.doTestES()
        self.assertTrue(mean_episode_reward >= 190)


if __name__ == "__main__":
    unittest.main(verbosity=2)
