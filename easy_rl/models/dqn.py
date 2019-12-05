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


class DQNModel(Model):
    """Implementation of Deep Q-Learning with all components of `Rainbow: Combining Improvements in Deep Reinforcement Learning`
    [Matteo Hessel et al., 2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/viewFile/17204/16680)"""

    def __init__(self, obs_space, action_space, scope="dqn_model", **kwargs):

        _default_config = {

            # network config
            # specify the parameters of network for each input state.
            # a default config will be used if empty
            # * it is recommended to overrider the `encode_states` function and then implement a custom network structure.
            "network_spec": {},

            # model config
            # Number of atoms for representing the distribution of return. When
            # this is greater than 1, distributional Q-learning is used.
            # the discrete supports are bounded by v_min and v_max
            # `A Distributional Perspective on Reinforcement Learning`[Marc G. Bellemare et al., 2017]
            # (https://arxiv.org/pdf/1707.06887.pdf)
            "num_atoms": 1,
            "v_min": -10.0,
            "v_max": 10.0,

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

            # whether to use parameter_noise for exploration
            # `Parameter Space Noise for Exploration`[Matthias Plappert et al., 2017]
            # (https://arxiv.org/abs/1706.01905)
            "parameter_noise": False,

            # scale of the parameter space noise
            "parameter_noise_scale": 0.1,

            # training config
            # whether to use huber_loss
            "use_huber_loss": False,

            # discount factor for accumulative reward
            "gamma": 0.99,

            # parameter for grad clipping
            "global_norm_clip": 40,

            # initialization of learning rate
            "init_lr": 0.001,

            # strategy of learning rate
            "lr_strategy_spec": {}
        }

        self.config = copy.deepcopy(_default_config)
        self.config.update(kwargs.get('model_config', {}))

        super(DQNModel, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            scope=scope,
            **kwargs)

    @property
    def perturbed_actions(self):
        return self.perturbed_actions_op

    @property
    def sync_target(self):
        """return operation for synchronizing the weights between source model and target model"""
        return self.sync_target_op

    @property
    def perturb_and_update(self):
        """return operation for reset the parameters of perturbed network and update noise scale"""
        return self.perturb_and_update_op

    @property
    def extra_act_fetches(self):
        return {"perturb_and_update": self.perturb_and_update_op}

    @property
    def extra_act_feed(self):
        """return the placeholder of dict for acting input:
            - eps: epsilon for exploration, to what probability the agent take a uniform-random action.
            - noise_kl_threshold: the bound of params_noise.
            - reset: reset the perturbed params with the latest scale of params_noise.
        """
        return {
            "eps": self.eps_ph,
            "noise_kl_threshold": self.noise_kl_threshold_ph,
            "reset": self.reset_ph,
            "update": self.update_ph
        }

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

    def _reset_and_update_perturb_params(self, action_logits,
                                         perturbed_action_logits,
                                         noise_kl_threshold, update, reset):
        """Implementation of parameter space noise refer to `Parameter Space Noise for Exploration`[Plappert et al., 2017]
        (https://arxiv.org/abs/1706.01905).
        update the scale of params noise, refresh the kl threshold and reset the params of perturb network

        Arguments:
            action_logits: logits of output action.
            perturbed_action_logits: logits of output action with parameter space noise.
            noise_kl_threshold: the desired threshold for KL-divergence between non-perturbed and perturbed policy.
            update: whether to update the adaptive noise scale.
            reset: whether to reset the perturbed policy with adaptive noise scale.

        Returns:
            reset the parameters of perturb network with given scale of noise and update the scale according to
            the input signal
        """

        def perturb_params():
            source_var_list = self.model_vars
            perturbed_var_list = self.perturb_model_vars
            assert len(source_var_list) == len(perturbed_var_list), "length of variable list is mismatched" \
                                                                    "source_vars:{}, perturbed_vars:{}".format(
                len(source_var_list), len(perturbed_var_list))
            ops = []
            for sv, pv in zip(source_var_list, perturbed_var_list):
                ops.append(
                    tf.assign(
                        pv, sv + tf.random_normal(
                            shape=tf.shape(sv),
                            stddev=self.perturbation_scale)))
            return tf.group(*ops)

        def update_perturb_scale():
            kl = tf.reduce_sum(
                tf.nn.softmax(action_logits) *
                (tf.nn.log_softmax(action_logits) -
                 tf.nn.log_softmax(perturbed_action_logits)),
                axis=-1)
            update_scale = tf.cond(
                tf.reduce_mean(kl) < noise_kl_threshold,
                lambda: self.perturbation_scale.assign(self.perturbation_scale * 1.01),
                lambda: self.perturbation_scale.assign(self.perturbation_scale / 1.01)
            )
            return update_scale

        _reset = tf.cond(reset, perturb_params, lambda: tf.group(*[]))
        with tf.control_dependencies([_reset]):
            _update = tf.cond(
                update, update_perturb_scale,
                lambda: tf.Variable(0., trainable=False, dtype=tf.float32))

        return _update

    @property
    def learn_feed(self):
        return OrderedDict(
            [("obs", self.obs_ph), ("actions", self.actions_ph),
             ("rewards", self.rewards_ph), ("next_obs", self.next_obs_ph),
             ("dones", self.dones_ph), ("weights",
                                        self.importance_weights_ph)])

    def _build_loss(self, obs_embedding, q_value_or_dist,
                    next_selected_q_value_or_dist, actions, dones, rewards,
                    importance_weights):
        """build q_value loss according to the bellman equation

        Arguments:
            obs_embedding: embedding for observation.
            q_value_or_dist: q_value or distribution of q_value of observation.
            next_selected_q_value_or_dist: q_value or distribution of q_value with specific action of next observation.
            actions: actions taken by agent.
            dones: terminal signals.
            rewards: immediate reward.
            importance_weights: importance weights for instances.
        """

        num_atoms = self.config.get('num_atoms', 1)
        gamma = self.config.get('gamma', 0.99)
        n_step = self.config.get('n_step', 1)

        if num_atoms > 1:
            """Distributional Q-learning"""

            v_min = self.config.get('v_min', -10)
            v_max = self.config.get('v_max', 10)

            # fixed support position
            delta_z = float(v_max - v_min) / (num_atoms - 1)
            z = v_min + tf.range(num_atoms, dtype=tf.float32) * delta_z

            # project distribution: z(s`, a*) -> r + gamma * z(s`, a*)
            # [batch_size, 1] * [1, num_atoms] = [batch_size, num_atoms]
            rewards = tf.expand_dims(rewards, -1)
            support_z = rewards + (gamma**n_step) * tf.expand_dims(
                1.0 - dones, -1) * tf.expand_dims(z, 0)
            support_z = tf.clip_by_value(support_z, v_min, v_max)
            b = (support_z - v_min) / delta_z
            lb = tf.cast(tf.floor(b), tf.int32)
            ub = tf.cast(tf.ceil(b), tf.int32)

            # in case b is an integer, lb = ub = b
            is_equal = tf.to_float(tf.equal(lb, ub))
            u_minus_b, b_minus_l = tf.cast(
                ub, tf.float32) - b + is_equal, b - tf.cast(lb, tf.float32)

            # q_value_or_dist: [batch_size, action_dim, num_atoms]
            # actons: [batch_size, action_dim, 1]
            # output: [batch_size, num_atoms]
            selected_action_prob = tf.reduce_sum(
                tf.reshape(q_value_or_dist, [-1, self.action_dim, num_atoms]) *
                tf.expand_dims(tf.one_hot(actions, self.action_dim), -1),
                axis=1)

            # calculate pj(x`, a*)(ub-b) and pj(x`, a*)(b-lb)
            # selected_action_prob: [batch_size, num_atoms]
            # u_minus_b: [batch_size, num_atoms]
            mj_u = next_selected_q_value_or_dist * b_minus_l
            mj_l = next_selected_q_value_or_dist * u_minus_b

            # project mj_* -> mi
            # project_matrix: [batch_size, num_atoms, num_atoms]
            project_matrix_l = tf.one_hot(lb, depth=num_atoms)
            mi_l = project_matrix_l * tf.expand_dims(mj_l, -1)

            project_matrix_u = tf.one_hot(ub, depth=num_atoms)
            mi_u = project_matrix_u * tf.expand_dims(mj_u, -1)

            mi = tf.reduce_sum(mi_l + mi_u, axis=1)
            mi = tf.stop_gradient(mi)

            self.td_error = tf.reduce_sum(
                -mi * tf.log(selected_action_prob), axis=-1)

        else:
            q_selected_value = tf.reduce_sum(
                obs_embedding * tf.one_hot(actions, depth=self.action_dim),
                axis=1)

            target_q_value = tf.stop_gradient(rewards +
                                              next_selected_q_value_or_dist *
                                              (1.0 - dones) * (gamma**n_step))
            self.summary_ops["train"].append(
                tf.summary.histogram("q_value", q_selected_value))
            self.summary_ops["train"].append(
                tf.summary.histogram("target_q_value", target_q_value))
            self.td_error = tf.squeeze(q_selected_value - target_q_value)

        use_huber_loss = self.config.get('use_huber_loss', False)

        if use_huber_loss and num_atoms == 1:
            # no need to use huber loss over a cross-entropy error
            loss_op = tf.reduce_mean(
                importance_weights * self._huber_loss(self.td_error))
        elif num_atoms == 1:
            loss_op = tf.reduce_mean(
                importance_weights * tf.square(self.td_error))
        else:
            loss_op = tf.reduce_mean(
                importance_weights * tf.square(self.td_error))
        self.summary_ops["train"].append(
            tf.summary.histogram("td_error", self.td_error))
        self.summary_ops["train"].append(tf.summary.scalar("loss", loss_op))

        return loss_op

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
            num_atoms = self.config.get('num_atoms', 1)
            v_min = self.config.get('v_min', -10)
            v_max = self.config.get('v_max', 10)

            self.obs_embedding = self._encode_obs(
                input_obs=self.preprocessed_obs_ph, scope="source_model")
            self.model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/source_model")

            params_noise = self.config.get('parameter_noise', False)
            if params_noise:
                init_perturbation_scale = self.config.get(
                    'perturbation_scale', 0.1)
                self.perturbation_scale = tf.get_variable(
                    'perturbation_scale',
                    dtype=tf.float32,
                    initializer=init_perturbation_scale)

                # build a perturb network for parameter space noise
                self.perturb_obs_embedding = self._encode_obs(
                    input_obs=self.preprocessed_obs_ph, scope="perturb_model")
                self.perturb_model_vars = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES,
                    scope=tf.get_variable_scope().name + "/perturb_model")

            self.next_obs_embedding = self._encode_obs(
                input_obs=self.preprocessed_next_obs_ph, scope="target_model")
            self.target_model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/target_model")
            if num_atoms > 1:
                # Distributional Q-Learning
                # fixed support position
                delta_z = float(v_max - v_min) / (num_atoms - 1)
                z = v_min + tf.range(num_atoms, dtype=tf.float32) * delta_z

                if dueling:
                    assert len(self.obs_embedding) == 2, "Invalid number of parameters,`action advantage` " \
                                                            "and `state value` is expected"
                    support_logits, value_preds = self.obs_embedding
                    next_support_logits, next_value_preds = self.next_obs_embedding

                    def _get_q_dist_value_and_prob(support_logits_,
                                                   value_preds_, action_dim_):
                        support_logits_ = tf.reshape(
                            support_logits_, [-1, action_dim_, num_atoms])
                        support_logits_mean = tf.reduce_mean(
                            support_logits_, axis=1, keep_dims=True)
                        support_logit_per_action = support_logits_ - support_logits_mean + value_preds_

                        # get q_value and q_dist_prob
                        q_dist_prob = tf.nn.softmax(support_logit_per_action)
                        obs_embedding = tf.reduce_sum(z * q_dist_prob, axis=-1)

                        return obs_embedding, q_dist_prob

                    self.obs_embedding, self.q_dist_prob = _get_q_dist_value_and_prob(
                        support_logits, value_preds, self.action_dim)

                    self.next_obs_embedding, self.next_q_dist_prob = _get_q_dist_value_and_prob(
                        next_support_logits, next_value_preds, self.action_dim)

                    if params_noise:
                        perturb_support_logits, perturb_value_preds = self.perturb_obs_embedding
                        self.perturb_obs_embedding, self.perturb_q_dist_prob = _get_q_dist_value_and_prob(
                            perturb_support_logits, perturb_value_preds,
                            self.action_dim)

                else:
                    # get q_value and q_dist_prob
                    support_logit_per_action = tf.reshape(
                        self.obs_embedding, [-1, self.action_dim, num_atoms])
                    # z: [num_atoms]
                    # support_logit_per_action: [batch_size, action_dim, num_atoms]
                    # output: [batch_size, action_dim, num_atoms]
                    self.q_dist_prob = tf.nn.softmax(support_logit_per_action)
                    self.obs_embedding = tf.reduce_sum(
                        z * self.q_dist_prob, axis=-1)

                    next_support_logit_per_action = tf.reshape(
                        self.next_obs_embedding,
                        [-1, self.action_dim, num_atoms])
                    self.next_q_dist_prob = tf.nn.softmax(
                        next_support_logit_per_action)
                    self.next_obs_embedding = tf.reduce_sum(
                        z * self.next_q_dist_prob, axis=-1)

                    if params_noise:
                        perturb_support_logit_per_action = tf.reshape(
                            self.perturb_obs_embedding,
                            [-1, self.action_dim, num_atoms])
                        perturb_q_dist_prob = tf.nn.softmax(
                            perturb_support_logit_per_action)
                        self.perturb_obs_embedding = tf.reduce_sum(
                            z * perturb_q_dist_prob, axis=-1)

            else:
                self.q_dist_prob = None
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

                    if params_noise:
                        perturb_action_score, perturb_state_value = self.perturb_obs_embedding
                        perturb_action_score_mean = tf.reduce_mean(
                            perturb_action_score, axis=1, keep_dims=True)
                        self.perturb_obs_embedding = perturb_action_score - perturb_action_score_mean + perturb_state_value
                else:
                    pass

            if double_q:
                q_selected_action = tf.argmax(self.obs_embedding, 1)
            else:
                q_selected_action = tf.argmax(self.next_obs_embedding, 1)

            # build output action_op
            self.action_op = self._actions_distribution(self.obs_embedding)

            if params_noise:
                # add parameter space noise
                self.perturbed_actions_op = self._actions_distribution(
                    self.perturb_obs_embedding)
                self.perturb_and_update_op = self._reset_and_update_perturb_params(
                    self.obs_embedding, self.perturb_obs_embedding,
                    self.noise_kl_threshold_ph, self.update_ph, self.reset_ph)
                self.summary_ops["train"].append(
                    tf.summary.scalar("noise_kl_threshold",
                                      self.noise_kl_threshold_ph))

            else:
                self.perturb_and_update_op = tf.no_op()

            self.sync_target_op = self._sync_target_params()

            if num_atoms > 1:
                # output: [batch_size, num_atoms]
                next_selected_q_value_or_dist = tf.reduce_sum(
                    self.next_q_dist_prob * tf.expand_dims(
                        tf.one_hot(q_selected_action, self.action_dim), -1),
                    axis=1)
            else:
                # output: [batch_size, 1]
                next_selected_q_value_or_dist = tf.reduce_sum(
                    self.next_obs_embedding * tf.one_hot(
                        q_selected_action, self.action_dim),
                    axis=1)

            # build loss_op
            self.loss_op = self._build_loss(
                obs_embedding=self.obs_embedding,
                q_value_or_dist=self.q_dist_prob,
                next_selected_q_value_or_dist=next_selected_q_value_or_dist,
                actions=self.actions_ph,
                rewards=self.rewards_ph,
                dones=self.dones_ph,
                importance_weights=self.importance_weights_ph)

            if kwargs.get("is_replica", False):
                self._scope_vars = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
                return

            # build train_op
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

    def _build_ph_op(self, obs_space, action_space):

        super(DQNModel, self)._build_ph_op(
            obs_space=obs_space, action_space=action_space)

        # add extra placeholder
        self.eps_ph = tf.placeholder_with_default(input=-1.0, shape=())
        self.importance_weights_ph = tf.placeholder(
            dtype=tf.float32, shape=[None])

        # parameter space noise
        self.noise_kl_threshold_ph = tf.placeholder_with_default(
            input=0.1, shape=())
        self.reset_ph = tf.placeholder_with_default(input=False, shape=())
        self.update_ph = tf.placeholder_with_default(input=True, shape=())

    def _huber_loss(self, x, delta=1.0):
        return tf.where(
            tf.abs(x) < delta,
            tf.square(x) * 0.5, delta * (tf.abs(x) - 0.5 * delta))
