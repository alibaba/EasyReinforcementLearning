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

import copy
from collections import OrderedDict
import tensorflow as tf

from easy_rl.models.model_base import Model
from easy_rl.utils import *


class DDPGModel(Model):
    """Deep Deterministic Policy Gradient agent.
    The piece de resistance of deep reinforcement learning as described by [Lillicrap, et al. (2015)]
    (https://arxiv.org/pdf/1509.02971v2.pdf).

    DDPG is a policy gradient algorithm that uses a stochastic behavior policy for good exploration but
    estimates a deterministic target policy"""

    def __init__(self, obs_space, action_space, scope="ddpg_model", **kwargs):

        _default_config = {

            # network config
            # specify the parameters of network for each input state.
            # a default config will be used if empty
            # * it is recommended to overrider the `encode_obs` function and then implement a custom network structure.
            "network_spec": {},

            # critic network config
            "critic_network_spec": {},

            # whether to use parameter_noise for exploration
            # Note that ornstein uhlenbeck noise is orthogonal with parameter space noise and
            # can be applied in action space to enhance the exploration
            # `Parameter Space Noise for Exploration`[Matthias Plappert et al., 2017]
            # (https://arxiv.org/abs/1706.01905)
            "parameter_noise": False,

            # scale of the parameter space noise
            "parameter_noise_scale": 0.1,

            # set configuration to use ornstein_uhlenbeck noise
            "ornstein_uhlenbeck_spec": {},

            # training config
            # whether to use huber_loss
            "use_huber_loss": False,

            # parameter for grad clipping
            "global_norm_clip": 40,

            # initialization of learning rate for actor-network
            "actor_lr_init": 0.001,

            # strategy of actor learning rate
            "actor_lr_strategy_spec": {},

            # initialization of learning rate for critic-network
            "critic_lr_init": 0.001,

            # strategy of critic learning rate
            "critic_lr_strategy_spec": {},

            # moving average of parameters.
            # 1.0 corresponds to hard synchronization
            "tau": 1.0,
        }

        self.config = copy.deepcopy(_default_config)
        self.config.update(kwargs.get('model_config', {}))

        super(DDPGModel, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            scope=scope,
            **kwargs)

    @property
    def perturbed_actions(self):
        """action with parameter space noise"""
        assert self.config.get(
            'parameter_noise',
            False), "Set parameter_noise to `True` to use perturbed_actions"
        return self.perturbed_actions_op

    @property
    def sync_target(self):
        """return operation for synchronizing the weights between source model and target model"""
        return self.sync_target_op

    @property
    def extra_act_fetches(self):
        return {"obs_embedding": self.obs_embedding}

    @property
    def extra_act_feed(self):
        """return the placeholder of dict for acting input:
            - params_noise_threshold: the bound of params_noise.
            - reset: reset the perturbed params with the latest scale of params_noise.
        """
        return {
            "perturbation_scale": self.perturbation_scale_ph,
            "reset": self.reset_ph
        }

    @property
    def extra_learn_fetches(self):
        """return loss"""
        return {
            "critic_loss": self.critic_loss_op,
            "actor_loss": self.actor_loss_op,
            "selected_q_value": self.selected_q_value,
            "td_error": self.td_error
        }

    @property
    def learn_feed(self):
        """return the placeholder of feed_dict for training input:
            - obs: observation obtained by agent.
            - action: action selected by agent.
            - target_obs: the next n step observation, mostly n=1.
            - reward: reward signal.
            - terminal: terminal signal.
        """

        return OrderedDict(
            [("obs", self.obs_ph), ("actions", self.actions_ph),
             ("rewards", self.rewards_ph), ("next_obs", self.next_obs_ph),
             ("dones", self.dones_ph), ("weights",
                                        self.importance_weights_ph)])

    @property
    def all_variables(self):
        return self._scope_vars

    @property
    def actor_sync_variables(self):
        return self.model_vars

    def _sync_target_params(self):
        """copy the parameters from source model to target model
        """
        assign_ops = []
        tau = self.config["tau"]
        for actor_source, actor_target in zip(self.actor_model_vars,
                                              self.target_actor_model_vars):
            assign_ops.append(
                tf.assign(actor_target,
                          tau * actor_source + (1 - tau) * actor_target))

        for critic_source, critic_target in zip(self.critic_model_vars,
                                                self.target_critic_model_vars):
            assign_ops.append(
                tf.assign(critic_target,
                          tau * critic_source + (1 - tau) * critic_target))

        return tf.group(*assign_ops)

    def _reset_and_update_perturb_params(self, action, perturb_action,
                                         noise_kl_threshold, update, reset):
        """Implementation of parameter space noise refer to `Parameter Space Noise for Exploration`[Plappert et al., 2017]
        (https://arxiv.org/abs/1706.01905).
        update the scale of params noise, refresh the kl threshold and reset the params of perturb network.

        Arguments:
            action: output action.
            perturb_action: output action with parameter space noise.
            noise_kl_threshold: the desired threshold for KL-divergence between non-perturbed and perturbed policy.
            update: whether to update the adaptive noise scale.
            reset: whether to reset the perturbed policy with adaptive noise scale.

        Returns:
            reset the parameters of perturb network with given scale of noise and update the scale according to
            the input signal.
        """

        def perturb_params():
            source_var_list = self.actor_model_vars
            perturbed_var_list = self.perturbed_actor_model_vars
            assert len(source_var_list) == len(perturbed_var_list), "length of variable list is mismatched" \
                                                                    "source_vars:{}, perturbed_vars:{}".format(
                len(source_var_list), len(perturbed_var_list))
            ops = []
            for sv, pv in zip(source_var_list, perturbed_var_list):
                ops.append(
                    tf.assign(
                        pv, sv + tf.random_normal(
                            shape=tf.shape(sv),
                            stddev=self.params_noise_scale)))
            return tf.group(*ops)

        def update_perturb_scale():
            kl = tf.reduce_sum(
                tf.nn.softmax(action) *
                (tf.log(tf.nn.softmax(action)) - tf.log(
                    tf.nn.softmax(perturb_action))),
                axis=-1)
            update_scale = tf.cond(
                tf.reduce_mean(kl) < noise_kl_threshold,
                lambda: self.params_noise_scale.assign(self.params_noise_scale * 1.01),
                lambda: self.params_noise_scale.assign(self.params_noise_scale / 1.01)
            )
            return update_scale

        _reset = tf.cond(reset, perturb_params, lambda: tf.group(*[]))
        with tf.control_dependencies([_reset]):
            _update = tf.cond(update, update_perturb_scale,
                              lambda: tf.Variable(0., trainable=False))

        return _update

    def _build_loss(self, q_value, selected_q_value, next_q_value, dones,
                    rewards, importance_weights):
        """implement specific loss function according to the optimization target

        Arguments:
            q_value: the value of the critic calculated from the action predicted by actor and the related observation.
            selected_q_value: the output of the critic calculated from the action selected by behavior policy.
            next_q_value: the value of the target-critic calculated from the next-observation and the related action.
            dones: terminal signal.
            rewards: reward per step.
            importance_weights: set different weights for each sample to impact training.
        """

        # critic loss
        gamma = self.config.get('gamma', 0.99)
        self.td_error = tf.squeeze(selected_q_value) - (
            rewards + gamma * tf.squeeze(next_q_value) * (1 - dones))
        self.summary_ops["train"].append(
            tf.summary.histogram("td_error", self.td_error))
        self.summary_ops["train"].append(
            tf.summary.histogram("step_reward", rewards))
        self.summary_ops["train"].append(
            tf.summary.histogram("q_value", selected_q_value))
        critic_loss = tf.reduce_mean(
            tf.square(self.td_error) * importance_weights)
        self.summary_ops["train"].append(
            tf.summary.scalar("critic_loss", critic_loss))

        # actor loss
        actor_loss = -tf.reduce_mean(q_value)
        self.summary_ops["train"].append(
            tf.summary.scalar("actor_loss", actor_loss))

        return critic_loss, actor_loss

    def _build_train(self,
                     critic_loss,
                     actor_loss,
                     critic_model_vars,
                     actor_model_vars,
                     critic_optimizer,
                     actor_optimizer=None,
                     global_step=None):

        # minimize critic_loss
        grads_and_vars = critic_optimizer.compute_gradients(
            loss=critic_loss, var_list=critic_model_vars)
        grads_and_vars = [(grad, var) for grad, var in grads_and_vars
                          if grad is not None]

        # apply grad clipping
        grads, vars = zip(*grads_and_vars)
        clipped_grads, _ = tf.clip_by_global_norm(
            grads, clip_norm=self.config.get('global_norm_clip', 40))
        grads_and_vars = list(zip(clipped_grads, vars))

        critic_train_op = critic_optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # minimize actor_loss
        with tf.control_dependencies([critic_train_op]):
            actor_optimizer_ = actor_optimizer or critic_optimizer
            grads_and_vars = actor_optimizer_.compute_gradients(
                loss=actor_loss, var_list=actor_model_vars)
            grads_and_vars = [(grad, var) for grad, var in grads_and_vars
                              if grad is not None]

            # apply grad clipping
            grads, vars = zip(*grads_and_vars)
            clipped_grads, _ = tf.clip_by_global_norm(
                grads, clip_norm=self.config.get('global_norm_clip', 40))
            grads_and_vars = list(zip(clipped_grads, vars))

            train_op = actor_optimizer_.apply_gradients(
                grads_and_vars, global_step=None)

        return train_op

    def _build_graph(self, scope, **kwargs):

        # get or create global_step
        self.global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):

            ornstein_uhlenbeck_spec = self.config.get(
                'ornstein_uhlenbeck_spec', {})
            if len(ornstein_uhlenbeck_spec) > 0:
                self.action_distribution_type = "Identity"
            self.preprocessed_obs_ph, self.preprocessed_next_obs_ph = self.preprocess_obs(
                self.obs_ph, self.next_obs_ph)
            # encode the input obs
            self.obs_embedding = self._encode_obs(
                input_obs=self.preprocessed_obs_ph, scope="encode_obs")
            self.actor_model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/encode_obs/")
            # maintain `log_std` or `ou_state` with independent variable
            self.action_log_std_or_ou_state = tf.get_variable(
                name="action_log_std_or_ou_state",
                shape=(self.action_dim, ),
                dtype=tf.float32,
                initializer=tf.zeros_initializer,
                trainable=(self.action_distribution_type is "DiagGaussian"))

            # encode the input obs-action
            action_mean = self.obs_embedding
            self.q_value = self._encode_obs_action(
                input_obs=self.preprocessed_obs_ph,
                input_action=action_mean,
                scope="encode_obs_action")
            self.critic_model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/encode_obs_action/")
            self.selected_q_value = self._encode_obs_action(
                input_obs=self.preprocessed_obs_ph,
                input_action=self.actions_ph,
                scope="encode_obs_action")
            self.model_vars = self.actor_model_vars + self.critic_model_vars

            params_noise = self.config.get('params_noise', False)
            # add perturbed model to use parameter space noise
            if params_noise:
                init_params_noise_scale = self.config.get(
                    'parameter_noise_scale', 0.1)
                self.params_noise_scale = tf.get_variable(
                    'params_noise_scale',
                    dtype=tf.float32,
                    initializer=init_params_noise_scale)
                # encode the input obs with perturbed params
                self.perturbed_obs_embedding = self._encode_obs(
                    input_obs=self.preprocessed_obs_ph,
                    scope="perturbed_encode_obs")

                self.perturbed_actor_model_vars = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name +
                    "/perturbed_encode_obs/")

            # target model
            # encode the input obs
            self.next_obs_embedding = self._encode_obs(
                input_obs=self.preprocessed_next_obs_ph,
                scope="target_encode_obs")
            # encode the input obs-action
            target_action_mean = self.next_obs_embedding
            self.next_q_value = self._encode_obs_action(
                input_obs=self.preprocessed_next_obs_ph,
                input_action=target_action_mean,
                scope="target_encode_obs_action")
            self.next_q_value = tf.stop_gradient(self.next_q_value)

            self.target_actor_model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/target_encode_obs/")
            self.target_critic_model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name +
                "/target_encode_obs_action/")

            self.target_model_vars = self.target_actor_model_vars + self.target_critic_model_vars

            # build output action_op
            action_params = (self.obs_embedding,
                             self.action_log_std_or_ou_state)
            self.action_op = self._actions_distribution(
                action_params, **ornstein_uhlenbeck_spec)

            if params_noise:
                perturbed_actions_params = (self.perturbed_obs_embedding,
                                            self.action_log_std_or_ou_state)

                self.perturbed_actions_op = self._actions_distribution(
                    perturbed_actions_params)
                self.perturb_and_update_op = self._reset_and_update_perturb_params(
                    self.action_op, self.perturbed_actions_op,
                    self.perturbation_scale_ph, self.update_ph, self.reset_ph)

            # synchronize parameters between source model and target model
            self.sync_target_op = self._sync_target_params()

            # build loss_op
            self.critic_loss_op, self.actor_loss_op = self._build_loss(
                q_value=self.q_value,
                selected_q_value=self.selected_q_value,
                next_q_value=self.next_q_value,
                dones=self.dones_ph,
                rewards=self.rewards_ph,
                importance_weights=self.importance_weights_ph)

            if kwargs.get("is_replica", False):
                self._scope_vars = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                return

            # build train_op
            actor_lr = self.config.get('actor_lr_init', 1e-3)
            actor_lr_strategy_spec = self.config.get('actor_lr_strategy_spec',
                                                     {})
            # apply different decay strategy of learning rate
            actor_lr = learning_rate_utils.LearningRateStrategy(
                init_lr=actor_lr,
                strategy_spec=actor_lr_strategy_spec)(self.global_step)
            actor_opt = tf.train.AdamOptimizer(learning_rate=actor_lr)
            self.summary_ops["train"].append(
                tf.summary.scalar("actor_learning_rate", actor_lr))

            critic_lr = self.config.get('critic_lr_init', 1e-2)
            critic_lr_strategy_spec = self.config.get(
                'critic_lr_strategy_spec', {})
            # apply different decay strategy of learning rate
            critic_lr = learning_rate_utils.LearningRateStrategy(
                init_lr=critic_lr,
                strategy_spec=critic_lr_strategy_spec)(self.global_step)
            critic_opt = tf.train.AdamOptimizer(learning_rate=critic_lr)
            self.summary_ops["train"].append(
                tf.summary.scalar("critic_learning_rate", critic_lr))

            self.train_op = self._build_train(
                critic_loss=self.critic_loss_op,
                actor_loss=self.actor_loss_op,
                critic_model_vars=self.critic_model_vars,
                actor_model_vars=self.actor_model_vars,
                critic_optimizer=critic_opt,
                actor_optimizer=actor_opt,
                global_step=self.global_step)

            self._scope_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def _encode_obs(self, input_obs, scope="encode_obs"):
        """build network to encode input feature.the desired output is action

        Arguments:
          input_obs: the (list, dict)[of] input tensor of observation.
          scope: the name of variable scope
        """

        with tf.variable_scope(name_or_scope=scope):
            # override the function `build_graph` to implement the specific network manually

            if len(self.config.get('network_spec')) == 0:
                # must set action_dim to use default network
                action_dim = self.action_dim

                if isinstance(input_obs, dict):
                    is_multi_input = len(input_obs) > 1
                    is_image_input = any(
                        [(len(ob.shape) > 2) for _, ob in input_obs.items()])
                elif isinstance(input_obs, list):
                    is_multi_input = len(input_obs) > 1
                    is_image_input = any(
                        [(len(ob.shape) > 2) for ob in input_obs])
                else:
                    is_multi_input = False
                    is_image_input = len(input_obs.shape) > 2

                if is_image_input and is_multi_input:
                    raise ValueError(
                        "the default convolution network accepts only one input but {} given"
                        .format(len(input_obs)))
                if is_image_input and not is_multi_input:
                    obs_embedding = layer_utils.DefaultConvNetwork(
                        action_dim=action_dim)(input_obs)
                else:
                    obs_embedding = layer_utils.DefaultFCNetwork(
                        action_dim=action_dim)(input_obs)
            else:
                # build computation graph from configuration
                assert not isinstance(input_obs, list), "list type is forbidden, key for each channel of input_" \
                                                           "obs should be supplied to build graph from configuration" \
                                                           "with multi-channel data"
                obs_embedding = layer_utils.build_model(
                    inputs=input_obs,
                    network_spec=self.config['network_spec'],
                    is_training_ph=self.is_training_ph)

            return obs_embedding

    def _encode_obs_action(self,
                           input_obs,
                           input_action,
                           scope="encode_obs_action"):
        """build critic network to encode input obs-action. the desired output
        is a scalar.

        Arguments:
          input_obs: the (list, dict)[of] input tensor of observation.
          input_action: the input tensor of action.
          scope: the name of variable scope.
        """
        with tf.variable_scope(name_or_scope=scope):
            #===============================================================================
            # override the function to implement the specific network manually
            #===============================================================================
            if len(self.config.get('critic_network_spec')) == 0:
                # Note that the default config only support single-channel input of obs
                # must set action_dim to use default network
                action_dim = self.action_dim

                if isinstance(input_obs, dict):
                    is_multi_input = len(input_obs) > 1
                    is_image_input = any(
                        [(len(ob.shape) > 2) for _, ob in input_obs.items()])
                elif isinstance(input_obs, list):
                    is_multi_input = len(input_obs) > 1
                    is_image_input = any(
                        [(len(ob.shape) > 2) for ob in input_obs])
                else:
                    is_multi_input = False
                    is_image_input = len(input_obs.shape) > 2

                if is_image_input and is_multi_input:
                    raise ValueError(
                        "the default convolution network accepts only one input but {} given"
                        .format(len(input_obs)))
                if is_image_input and not is_multi_input:
                    obs_action_embedding = layer_utils.DefaultConvNetwork(
                        action_dim=action_dim,
                        input_action=input_action)(input_obs)
                else:
                    obs_action_embedding = layer_utils.DefaultFCNetwork(
                        action_dim=action_dim,
                        input_action=input_action)(input_obs)
            else:
                input_obs = {"input_obs": input_obs, "action": input_action}
                obs_action_embedding = layer_utils.build_model(
                    inputs=input_obs,
                    network_spec=self.config['critic_network_spec'],
                    is_training_ph=self.is_training_ph)

            return tf.squeeze(obs_action_embedding)

    def _build_ph_op(self, obs_space, action_space):

        super(DDPGModel, self)._build_ph_op(
            obs_space=obs_space, action_space=action_space)

        # add extra placeholder of
        self.importance_weights_ph = tf.placeholder(
            dtype=tf.float32, shape=[None])

        # parameter space noise
        self.perturbation_scale_ph = tf.placeholder_with_default(
            input=0.1, shape=())
        self.reset_ph = tf.placeholder_with_default(input=False, shape=())
        self.update_ph = tf.placeholder_with_default(input=True, shape=())
