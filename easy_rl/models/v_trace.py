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

import tensorflow as tf
from easy_rl.models.model_base import Model
import copy
from collections import OrderedDict
from easy_rl.utils import *


class VTraceModel(Model):
    """Implementation the v-trace algorithm for Importance Weighted Actor-Learner Architecture
    [Lasse Espeholt et al., 2017](https://arxiv.org/abs/1802.01561) """

    def __init__(self, obs_space, action_space, scope="vtrace_model",
                 **kwargs):

        _default_config = {

            # network config
            # specify the parameters of network for each input state.
            # a default config will be used if empty
            # * it is recommended to overrider the `encode_obs` function and then implement a custom network structure.
            "network_spec": {},

            # model config
            # clipping of Importance weighing for value-trace
            "rho_clipping": None,

            #clipping of Importance weigthing for gradient-trace
            "pg_rho_clipping": None,

            # training config
            # coefficient for entroy
            "entroy_coef": 1e-3,

            # coefficient for loss of value function
            "vf_coef": 0.5,

            # parameter for grad clipping
            "global_norm_clip": 40,

            # initialization of learning rate
            "lr_init": 0.001,
        }

        self.config = copy.deepcopy(_default_config)
        self.config.update(kwargs.get('model_config', {}))

        super(VTraceModel, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            scope=scope,
            **kwargs)

    @property
    def extra_act_fetches(self):
        """return the sample_action with action logits and state_value.
        """

        return {
            "logits": self.action_dist.distribution_params,
        }

    @property
    def learn_feed(self):
        """return the dict of placeholder of action logits and state_value.
        the input value will be used to calculate v-trace actor critic targets.
        """

        return OrderedDict([("obs", self.obs_ph), ("actions", self.actions_ph),
                            ("rewards", self.rewards_ph),
                            ("dones", self.dones_ph), ("logits",
                                                       self.logits_ph)])

    @property
    def extra_learn_fetches(self):
        """return extra fetches after call agent.learn"""
        return {"loss": self.loss_op}

    @property
    def all_variables(self):
        return self._scope_vars

    @property
    def actor_sync_variables(self):
        return self.model_vars

    def v_trace_estimation(self, value_preds, actions, dones, rewards, logits):
        """Calculates V-trace actor critic targets.

        Arguments:
            value_preds: state_value estimated by current policy.
                          Note that one more state_value is appended to value_preds.
            actions: action sampled by behaviour policy.
            dones: terminal signal.
            rewards:: immediate reward return by env.
            logits: value of logits given by behaviour policy.

        Returns:
            remedied value-target and state-action dependent estimator of advantage
        """
        discount = self.config.get('discount', 0.99)
        discounts = tf.to_float(~tf.cast(dones, tf.bool)) * discount

        with tf.device("/cpu:0"):
            behaviour_log_p = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=actions)
            target_log_p = self.action_dist.log_p(action=actions)

            log_rhos = target_log_p - behaviour_log_p
            log_rhos = log_rhos[:-1]
            rhos = tf.exp(log_rhos)

            rho_clipping = self.config.get('rho_clipping', None)
            if rho_clipping:
                clipped_rhos = tf.minimum(
                    tf.cast(rho_clipping, tf.float32), rhos)
            else:
                clipped_rhos = rhos

            pg_rho_clipping = self.config.get('pg_rho_clipping', None)
            if pg_rho_clipping:
                clipped_pg_rhos = tf.minimum(
                    tf.cast(pg_rho_clipping, tf.float32), rhos)
            else:
                clipped_pg_rhos = rhos

            cs = tf.minimum(1.0, rhos)

            next_state_value = value_preds[1:]
            state_value = value_preds[:-1]
            last_state_value = value_preds[-1]

            deltas = clipped_rhos * (
                rewards + discounts * next_state_value - state_value)

            # V-trace vs are calculated through a scan from the back to the beginning
            # of the given trajectory.
            sequences = (
                tf.reverse(discounts, axis=[0]),
                tf.reverse(cs, axis=[0]),
                tf.reverse(deltas, axis=[0]),
            )

            def scanfunc(acc, sequence_item):
                discount_t, c_t, delta_t = sequence_item
                return delta_t + discount_t * c_t * acc

            initial_values = tf.zeros_like(last_state_value)
            vs_minus_v_xs = tf.scan(
                fn=scanfunc,
                elems=sequences,
                initializer=initial_values,
                parallel_iterations=1,
                back_prop=False,
                name='scan')
            # Reverse the results back to original order.
            vs_minus_v_xs = tf.reverse(
                vs_minus_v_xs, [0], name='vs_minus_v_xs')
            # Add V(x_s) to get v_s.
            vs = tf.add(vs_minus_v_xs, state_value, name='vs')

            # Advantage for policy gradient.
            vs_t_plus_1 = tf.concat(
                [vs[1:], tf.expand_dims(last_state_value, 0)], axis=0)
            pg_advantages = (clipped_pg_rhos *
                             (rewards + discounts * vs_t_plus_1 - state_value))

            advantages = tf.stop_gradient(pg_advantages)
            value_target = tf.stop_gradient(vs)

            return value_target, advantages

    def _build_loss(self, value_preds, actions, rewards, dones, logits):
        """implement specific loss function according to the optimization target

        Arguments:
            value_preds: estimator of state value from the target policy(the policy to optimize)
            actions: action taken by the behavior policy.
            rewards: reward per step observed from the behavior policy.
            dones: terminal signal.
            logits: logits from the behavior policy.
        """

        value_target, advantages = self.v_trace_estimation(
            value_preds=value_preds,
            actions=actions,
            dones=dones[:-1],
            rewards=rewards[:-1],
            logits=logits)
        self.summary_ops["train"].append(
            tf.summary.histogram("value_target", value_target))
        self.summary_ops["train"].append(
            tf.summary.histogram("value_preds", value_preds))
        self.summary_ops["train"].append(
            tf.summary.histogram("advantage", advantages))

        # loss of policy
        log_p = self.action_dist.log_p(actions)
        log_p = log_p[:-1]
        loss_p = -tf.reduce_mean(log_p * advantages)

        # loss of value function
        loss_v = .5 * tf.reduce_mean(
            tf.square(value_target - value_preds[:-1]))
        self.summary_ops["train"].append(
            tf.summary.scalar("loss_value", loss_v))

        vf_coef = self.config.get('vf_coef', 0.5)
        entroy_coef = self.config.get('entroy_coef', 1e-6)

        # entroy regularization
        entroy = self.action_dist.entropy()
        self.summary_ops["train"].append(
            tf.summary.histogram("entroy", entroy))

        # total loss
        loss_op = loss_p + vf_coef * loss_v - entroy_coef * tf.reduce_mean(
            entroy)

        return loss_op

    def _build_train(self, loss, optimizer, vars=None, global_step=None):

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

            # encode the input obs
            self.preprocessed_obs_ph, self.preprocessed_next_obs_ph = self.preprocess_obs(
                self.obs_ph, self.next_obs_ph)
            self.obs_embedding, self.value_preds = self._encode_obs(
                input_obs=self.preprocessed_obs_ph, scope="encode_obs")
            self.model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/encode_obs")

            # build output action_op
            self.action_op = self._actions_distribution(self.obs_embedding)

            # build loss_op
            self.loss_op = self._build_loss(
                value_preds=self.value_preds,
                actions=self.actions_ph,
                rewards=self.rewards_ph,
                dones=self.dones_ph,
                logits=self.logits_ph)

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

            opt = tf.train.AdamOptimizer(learning_rate=lr)

            self.train_op = self._build_train(
                loss=self.loss_op, optimizer=opt, global_step=self.global_step)

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
                        action_dim=action_dim, need_v=True)(input_obs)
                else:
                    obs_embedding = layer_utils.DefaultFCNetwork(
                        action_dim=action_dim, need_v=True)(input_obs)
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

        super(VTraceModel, self)._build_ph_op(
            obs_space=obs_space, action_space=action_space)

        # add extra placeholder of logits for training
        self.logits_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.action_dim])
