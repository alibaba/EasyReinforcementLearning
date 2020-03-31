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
import numpy as np

from easy_rl.models.model_base import Model
from easy_rl.utils import *


class BatchDQNModel(Model):
    """Implementation of batch-constrained reinforcement learning
    [Scott Fujimoto et al., 2019](https://arxiv.org/pdf/1910.01708.pdf)"""

    def __init__(self, obs_space, action_space, scope="bcq_model", **kwargs):

        _default_config = {

            # network config
            # specify the parameters of network for each input state.
            # a default config will be used if empty
            # * it is recommended to overrider the `encode_states` function and then implement a custom network structure.
            "network_spec": {},

            # model config
            # threshold for clip action
            "ratio_threshold" : 0.3,

            # use multi-step learning (especially forward-view multi-step)
            # Note that reward needs to be accumulated if n_step > 1
            "n_step": 1,

            # whether to use dueling dqn
            # `Dueling Network Architectures for Deep Reinforcement Learning`[Ziyu Wang et al., 2015]
            # (https://arxiv.org/abs/1511.06581)
            "dueling": False,

            # whether to use double dqn
            # `Deep Reinforcement Learning with Double Q-learning`[Hado van Hasselt et al., 2015]
            # (https://arxiv.org/abs/1509.06461)
            "double_q": True,

            # training config
            # whether to use huber_loss
            "use_huber_loss": False,

            # discount factor for accumulative reward
            "gamma": 0.99,

            # parameter for grad clipping
            "global_norm_clip": 40,

            # initialization of learning rate
            "init_lr": 0.001,

            # initialization of learning rate of behavior clone
            "clone_init_lr": 0.001,

            # strategy of learning rate
            "lr_strategy_spec": {},

            # strategy of learning rate of behavior clone
            "clone_lr_strategy_spec": {},
        }

        self.config = copy.deepcopy(_default_config)
        self.config.update(kwargs.get('model_config', {}))

        super(BatchDQNModel, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            scope=scope,
            **kwargs)

    @property
    def sync_target(self):
        """return operation for synchronizing the weights between source model and target model"""
        return self.sync_target_op

    @property
    def extra_learn_fetches(self):
        """return the additional fetches when training
        """
        return {"loss": self.loss_op, "td_error": self.td_error}

    @property
    def all_variables(self):
        """return all variables of model object"""
        return self._scope_vars

    @property
    def actor_sync_variables(self):

        return self.model_vars

    def _sync_target_params(self):
        """copy the parameters from source model to target model
        """
        assign_ops = []
        for source, target in zip(self.model_vars, self.target_model_vars):
            assign_ops.append(tf.assign(target, source))

        return tf.group(*assign_ops)

    @property
    def learn_feed(self):
        return OrderedDict(
            [("obs", self.obs_ph), ("actions", self.actions_ph),
             ("rewards", self.rewards_ph), ("next_obs", self.next_obs_ph),
             ("dones", self.dones_ph), ("weights",
                                        self.importance_weights_ph)])

    @property
    def clone_learn_feed(self):
        return OrderedDict(
            [("obs", self.obs_ph), ("actions", self.actions_ph),
             ("rewards", self.rewards_ph), ("next_obs", self.next_obs_ph),
             ("dones", self.dones_ph), ("weights",
                                        self.importance_weights_ph)])

    def _build_loss(self, obs_embedding, next_selected_q_value_or_dist,
                    actions, dones, rewards, importance_weights):
        """build q_value loss according to the bellman equation

        Arguments:
            obs_embedding: embedding for observation.
            next_selected_q_value: q_value with specific action of next observation.
            actions: actions taken by agent.
            dones: terminal signals.
            rewards: immediate reward.
            importance_weights: importance weights for instances.
        """

        gamma = self.config.get('gamma', 0.99)
        n_step = self.config.get('n_step', 1)

        q_selected_value = tf.reduce_sum(
            obs_embedding * tf.one_hot(actions, depth=self.action_dim),
            axis=1)

        target_q_value = tf.stop_gradient(rewards +
                                          next_selected_q_value_or_dist *
                                          (1.0 - dones) * (gamma**n_step))
        self.summary_ops["train"].append(
            tf.summary.scalar("dones", tf.reduce_mean(dones))
        )
        print("gamma", gamma, "n_step", n_step)
        self.summary_ops["train"].append(
            tf.summary.scalar("rewards", tf.reduce_mean(rewards))
        )
        self.summary_ops["train"].append(
            tf.summary.scalar("next_q_value_s", tf.reduce_mean(next_selected_q_value_or_dist))
        )
        self.summary_ops["train"].append(
            tf.summary.scalar("target_q_value_s", tf.reduce_mean(target_q_value))
        )
        self.summary_ops["train"].append(
            tf.summary.scalar("q_value_s", tf.reduce_mean(q_selected_value))
        )
        self.summary_ops["train"].append(
            tf.summary.histogram("q_value", q_selected_value))
        self.summary_ops["train"].append(
            tf.summary.histogram("target_q_value", target_q_value))
        self.td_error = tf.squeeze(q_selected_value - target_q_value)

        use_huber_loss = self.config.get('use_huber_loss', False)

        if use_huber_loss:
            # no need to use huber loss over a cross-entropy error
            loss_op = tf.reduce_mean(
                importance_weights * self._huber_loss(self.td_error))
        else:
            loss_op = tf.reduce_mean(
                importance_weights * tf.square(self.td_error))

        self.summary_ops["train"].append(
            tf.summary.histogram("td_error", self.td_error))

        self.regularization_loss = [reg_loss for reg_loss in tf.losses.get_regularization_losses() if "source_model" in reg_loss.name]
        if len(self.regularization_loss) > 0:
            regularization_loss = tf.reduce_mean(self.regularization_loss)
            loss_op += regularization_loss
            self.summary_ops["train"].append(tf.summary.scalar("regularization_loss", regularization_loss))

        self.summary_ops["train"].append(tf.summary.scalar("loss", loss_op))

        return loss_op

    def _build_clone_loss(self, action_logits, actions):
        """
        create the loss to learning behavior clone

        Arguments:
            action_logits: logits of action estimated by model
            actions: truly action taken by behavioral policy
        """

        clone_loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=action_logits,
            labels=actions
            ))

        self.clone_regularization_loss = [reg_loss for reg_loss in tf.losses.get_regularization_losses() if
                                    "generative_model" in reg_loss.name]
        if len(self.clone_regularization_loss) > 0:
            regularization_loss = tf.reduce_mean(self.clone_regularization_loss)
            clone_loss_op += regularization_loss
            self.summary_ops["clone_train"].append(tf.summary.scalar("clone_model_reg_loss", regularization_loss))

        self.summary_ops["clone_train"].append(tf.summary.scalar("clone_loss", clone_loss_op))

        return clone_loss_op

    def _build_train(self, loss, optimizer, vars, global_step=None):
        """
        construct the operation for optimization.

        Arguments:
            loss: the object loss function to minimize
            optimizer: optimizer to implement the optimization
            vars: the available variables to optimize
            global_step: record to total number of optimization
        """

        # compute gradients
        grads_and_vars = optimizer.compute_gradients(loss=loss, var_list=vars)
        grads_and_vars = [(grad, var) for grad, var in grads_and_vars
                          if grad is not None]

        # apply grad clipping
        grads, vars = zip(*grads_and_vars)
        clipped_grads, _ = tf.clip_by_global_norm(
            grads, clip_norm=self.config.get('global_norm_clip', 40))
        grads_and_vars = list(zip(clipped_grads, vars))

        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)
        return train_op

    def _build_graph(self, scope, **kwargs):

        # get or create global_step
        self.global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            self.preprocessed_obs_ph, self.preprocessed_next_obs_ph = self.preprocess_obs(
                self.obs_ph, self.next_obs_ph)

            # source model, encode the input obs
            dueling = self.config.get('dueling', False)
            double_q = self.config.get('double_q', False)

            self.obs_embedding = self._encode_obs(
                input_obs=self.preprocessed_obs_ph, scope="source_model")
            self.model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/source_model")

            self.g_embedding = self._generative_model(
                input_obs=self.preprocessed_obs_ph, scope="generative_model"
            )
            self.generative_model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/generative_model")

            self.next_obs_embedding = self._encode_obs(
                input_obs=self.preprocessed_next_obs_ph, scope="target_model")
            self.target_model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/target_model")

            if dueling:
                assert len(self.obs_embedding) == 2, "Invalid number of parameters,`action advantage` " \
                                                        "and `state value` is expected"
                action_score, value_preds = self.obs_embedding
                next_action_score, next_value_preds = self.next_obs_embedding

                action_score_mean = tf.reduce_mean(
                    action_score, axis=1, keep_dims=True)
                self.obs_embedding = action_score - action_score_mean + tf.expand_dims(
                    value_preds, -1)

                next_action_score_mean = tf.reduce_mean(
                    next_action_score, axis=1, keep_dims=True)
                self.next_obs_embedding = next_action_score - next_action_score_mean + tf.expand_dims(
                    next_value_preds, -1)

            if double_q:
                self.g_prob = tf.stop_gradient(tf.nn.softmax(self.g_embedding)+tf.constant(1e-10, dtype=tf.float32))
                max_prob = tf.reduce_max(self.g_prob, axis=-1, keepdims=True)
                action_mask = tf.cast(tf.less_equal(
                    self.g_prob / max_prob, self.ratio_threshold_ph), tf.float32) * tf.constant(-np.inf, dtype=tf.float32)
                q_selected_action = tf.argmax((self.obs_embedding + action_mask), 1)
            else:
                self.g_prob = tf.stop_gradient(tf.nn.softmax(self.g_embedding)+tf.constant(1e-10, dtype=tf.float32))
                max_prob = tf.reduce_max(self.g_prob, axis=-1, keepdims=True)
                action_mask = tf.cast(tf.less_equal(
                    self.g_prob / max_prob, self.ratio_threshold_ph), tf.float32) * tf.constant(-np.inf, dtype=tf.float32)
                q_selected_action = tf.argmax((self.next_obs_embedding + action_mask), 1)

            # build output action_op
            self.action_op = self._actions_distribution(self.obs_embedding)

            self.sync_target_op = self._sync_target_params()

            # output: [batch_size, 1]
            next_selected_q_value_or_dist = tf.reduce_sum(
                self.next_obs_embedding * tf.one_hot(
                    q_selected_action, self.action_dim),
                axis=1)

            # build loss of generative model
            self.clone_loss_op = self._build_clone_loss(
                action_logits=self.g_embedding,
                actions=self.actions_ph
            )

            # IPS evaluation
            self.ips_ratio = self.importance_ratio(
                self.obs_embedding, self.g_embedding, self.actions_ph)

            # build loss_op
            self.loss_op = self._build_loss(
                obs_embedding=self.obs_embedding,
                next_selected_q_value_or_dist=next_selected_q_value_or_dist,
                actions=self.actions_ph,
                rewards=self.rewards_ph,
                dones=self.dones_ph,
                importance_weights=self.importance_weights_ph)

            if kwargs.get("is_replica", False):
                self._scope_vars = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                return

            # build behavior clone train_op
            clon_init_lr = self.config.get('clone_init_lr', 1e-3)
            clone_lr_strategy_spec = self.config.get('clone_lr_strategy_spec', {})
            # apply different decay strategy of learning rate
            lr = learning_rate_utils.LearningRateStrategy(
                init_lr=clon_init_lr,
                strategy_spec=clone_lr_strategy_spec)(self.global_step)
            self.summary_ops["clone_train"].append(
                tf.summary.scalar("clone_learning_rate", lr))

            clone_opt = tf.train.AdamOptimizer(learning_rate=lr)

            self.clone_train_op = self._build_train(
                loss=self.clone_loss_op,
                optimizer=clone_opt,
                vars=self.generative_model_vars,
                global_step=self.global_step)

            # build train_op
            self.reset_global_step = tf.assign(self.global_step, tf.constant(0, tf.int64))

            init_lr = self.config.get('init_lr', 1e-3)
            lr_strategy_spec = self.config.get('lr_strategy_spec', {})

            # apply different decay strategy of learning rate
            lr = learning_rate_utils.LearningRateStrategy(
                init_lr=init_lr,
                strategy_spec=lr_strategy_spec)(self.global_step)
            self.summary_ops["train"].append(
                tf.summary.scalar("learning_rate", lr))

            opt = tf.train.AdamOptimizer(learning_rate=lr)

            self.train_op = self._build_train(
                loss=self.loss_op,
                optimizer=opt,
                vars=self.model_vars,
                global_step=self.global_step)

            self._scope_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def _encode_obs(self, input_obs, scope="encode_obs"):
        """build network to encode input feature.the desired output
        consists of two parts: action_logits and value_estimator

        Arguments:
          input_obs: the (list, dict)[of] input tensor of observation.
          scope: the name of variable scope
        """

        with tf.variable_scope(name_or_scope=scope):
            # override the function `build_graph` to implement the specific network manually

            if len(self.config.get('network_spec')) == 0:
                # must set action_dim to use default network
                action_dim = self.action_dim * self.config.get('num_atoms', 1)

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

                need_v = self.config.get('dueling', False)
                if is_image_input and not is_multi_input:
                    obs_embedding = layer_utils.DefaultConvNetwork(
                        action_dim=action_dim, need_v=need_v)(input_obs)
                else:
                    obs_embedding = layer_utils.DefaultFCNetwork(
                        action_dim=action_dim, need_v=need_v)(input_obs)
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

    def _generative_model(self, input_obs, scope="generative_model"):
        """build a behavioral cloning network to learn from offline batch data.

        Arguments:
            input_obs: the (list, dict)[of] input tensor of observation.
            scope: the name of variable scope.
        """
        raise NotImplementedError

    def importance_ratio(self, pai, estimated_pai, action):
        """calculate the importance_ratio for policy evaluation.

        Arguments:
            pai: the target policy.
            estimated_pai: estimation of behavior policy.
            action: action taken by behavior policy.
        """
        target_prob = tf.reduce_sum(pai * tf.one_hot(action, self.action_dim), axis=1)
        behavior_prob = tf.reduce_sum(estimated_pai * tf.one_hot(action, self.action_dim), axis=1)
        ratio = tf.clip_by_value(target_prob / behavior_prob, 1.0e-2, 1.0e2)

        return ratio

    def _build_ph_op(self, obs_space, action_space):

        super(BatchDQNModel, self)._build_ph_op(
            obs_space=obs_space, action_space=action_space)

        # add extra placeholder
        self.importance_weights_ph = tf.placeholder(
            dtype=tf.float32, shape=[None])
        self.ratio_threshold_ph = tf.placeholder_with_default(
            input=self.config.get("ratio_threshold", 0.3), shape=())

    def _huber_loss(self, x, delta=1.0):
        return tf.where(
            tf.abs(x) < delta,
            tf.square(x) * 0.5, delta * (tf.abs(x) - 0.5 * delta))

    def add_extra_summary_op(self):
        super(BatchDQNModel, self).add_extra_summary_op()

        self.ips_score_op = tf.placeholder_with_default(
            tf.constant(0.0, dtype=tf.float32),
            shape=(),
            name="ips_score")
        self.ips_score_stepwise_op = tf.placeholder_with_default(
            tf.constant(0.0, dtype=tf.float32),
            shape=(),
            name="ips_score_stepwise")
        self.wnorm_ips_score_op = tf.placeholder_with_default(
            tf.constant(0.0, dtype=tf.float32),
            shape=(),
            name="wnorm_ips_score")
        self.wnorm_ips_score_stepwise_op = tf.placeholder_with_default(
            tf.constant(0.0, dtype=tf.float32),
            shape=(),
            name="wnorm_ips_score_stepwise")
        self.wnorm_ips_score_stepwise_mean_op = tf.placeholder_with_default(
            tf.constant(0.0, dtype=tf.float32),
            shape=(),
            name="wnorm_ips_score_stepwise_mean")

        self.summary_ops["extra"].extend([
            tf.summary.scalar(
                name="ips_score",
                tensor=self.ips_score_op),
            tf.summary.scalar(
                name="ips_score_stepwise",
                tensor=self.ips_score_stepwise_op),
            tf.summary.scalar(
                name="wnorm_ips_score",
                tensor=self.wnorm_ips_score_op),
            tf.summary.scalar(
                name="wnorm_ips_score_stepwise",
                tensor=self.wnorm_ips_score_stepwise_op),
            tf.summary.scalar(
                name="wnorm_ips_score_stepwise_mean",
                tensor=self.wnorm_ips_score_stepwise_mean_op),
        ])
