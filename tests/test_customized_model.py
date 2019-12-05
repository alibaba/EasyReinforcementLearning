from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
import numpy as np
import tensorflow as tf
import gym

from easy_rl.utils import layer_utils
from easy_rl.agents import agents
from easy_rl.models import DQNModel, PPOModel
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
    global_norm_clip=40,
    network_spec=[[
        {
            "inputs": "input_ph"
        },
        {
            "type": "dense",
            "units": 10,
            "use_bias": False,
            "activation": "relu",
            "kernel_initializer": 1.0
        },
        {
            "outputs": "layer0_output"
        },
    ], [
        {
            "inputs": "input_ph"
        },
        {
            "type": "dense",
            "units": 10,
            "use_bias": False,
            "activation": "relu",
            "kernel_initializer": 1.0
        },
        {
            "outputs": "layer1_output"
        },
    ], [
        {
            "inputs": ["layer0_output", "layer1_output"],
            "process_type": None
        },
        {
            "type": "dense",
            "units": 10,
            "use_bias": True,
            "activation": "relu"
        },
        {
            "outputs": ["layer2_output_0", "layer2_output_1"]
        },
    ], [
        {
            "inputs": "layer2_output_0"
        },
        {
            "type": "dense",
            "units": 1,
            "use_bias": False,
            "activation": None,
            "kernel_initializer": -1.0
        },
        {
            "outputs": "action_logits"
        },
    ], [{
        "inputs": "layer2_output_1"
    }, {
        "type": "dense",
        "units": 1,
        "use_bias": False,
        "activation": None,
        "kernel_initializer": 1.0
    }, {
        "outputs": "state_value"
    }]],
)

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


class CustomizedModelTest(unittest.TestCase):
    """Run commonly used algorithms with customized models.
    """

    def doTestDQN(self):
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
                        units=2 * DQN_MODEL_CONFIG["num_atoms"],
                        activation=None,
                        kernel_initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=0.01, seed=0))
                    return logits

        env = gym.make("CartPole-v0")
        dqn_g = tf.Graph()
        with dqn_g.as_default():
            agent = agents[DQN_AGENT_CONFIG["type"]](
                env.observation_space,
                env.action_space,
                DQN_AGENT_CONFIG,
                DQN_MODEL_CONFIG,
                distributed_spec={},
                custom_model=MyDQNModel)
        ob = env.reset()
        action, results = agent.act(
            [ob], deterministic=False, use_perturbed_action=False)
        next_ob, reward, done, info = env.step(action[0])

    def testDQN(self):
        try:
            self.doTestDQN()
        except Exception as ex:
            self.fail("doTestDQN raised {} unexpectedly!".format(Exception))
        finally:
            pass

    def doTestPPO(self):
        class MyPPOModel(PPOModel):
            def _encode_obs(self, input_obs, scope="encode_obs"):
                with tf.variable_scope(name_or_scope=scope):
                    outputs = layer_utils.build_model(
                        inputs=input_obs,
                        network_spec=self.config.get('network_spec'),
                        is_training_ph=self.is_training_ph)
                    return outputs

        env = gym.make("CartPole-v0")
        ppo_g = tf.Graph()
        with ppo_g.as_default():
            agent = agents[PPO_AGENT_CONFIG["type"]](
                env.observation_space,
                env.action_space,
                PPO_AGENT_CONFIG,
                PPO_MODEL_CONFIG,
                distributed_spec={},
                custom_model=MyPPOModel)

        ob = env.reset()
        action, results = agent.act([ob], False)
        next_ob, reward, done, info = env.step(action[0])

    def testPPO(self):
        try:
            self.doTestPPO()
        except Exception as ex:
            self.fail("doTestPPO raised {} unexpectedly!".format(Exception))
        finally:
            pass


if __name__ == "__main__":
    unittest.main(verbosity=2)
