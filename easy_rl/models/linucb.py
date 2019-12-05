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
import numpy as np
import tensorflow as tf

from easy_rl.models.model_base import Model


class LinUCBModel(Model):
    """Implementation of LinUCB algorithm
    """

    def __init__(self, obs_space, action_space, scope="linucb_model",
                 **kwargs):
        _default_config = {
            # \lambda balances the prior and likelihood
            "lbd": 1.0,
        }
        self.config = copy.deepcopy(_default_config)
        super(LinUCBModel, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            scope=scope,
            **kwargs)

    @property
    def learn_feed(self):
        return OrderedDict([("obs", self.obs_ph), ("actions", self.actions_ph),
                            ("rewards", self.rewards_ph)])

    @property
    def extra_learn_fetches(self):
        """return the additional fetches when training
        """
        return {
            "loss": self.loss_op,
        }

    def preprocess(self, obs):
        """Augment a constant one component.
        """
        obs = super(LinUCBModel, self).preprocess(obs)
        dummy_components = tf.ones_like(obs)
        dummy_component = tf.reduce_mean(dummy_components, 1)
        return tf.concat([obs, dummy_component], 1)

    def _build_graph(self, scope, **kwargs):
        self.global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            preprocessed_obs, _ = self.preprocess_obs(self.obs_ph)

            estimated_mean, estimated_stddev = self._encode_obs(
                input_obs=preprocessed_obs, scope="ridge_regression")
            # exploration via UCB
            self.action_op = tf.argmax(
                estimated_mean + estimated_stddev, -1, output_type=tf.int32)

            self.loss_op = self._build_loss(
                estimated_values=estimated_mean,
                selected_arms=self.actions_ph,
                target_values=self.rewards_ph)

            self.train_op = self._build_train(
                input_obs=preprocessed_obs,
                estimated_values=estimated_mean,
                selected_arms=self.actions_ph,
                target_values=self.rewards_ph,
                global_step=self.global_step)

            self._scope_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    def _encode_obs(self, input_obs, scope="encode_obs"):
        with tf.variable_scope(name_or_scope=scope):
            w = tf.get_variable(
                name="w",
                shape=(self.action_dim, input_obs.shape[1]),
                dtype=tf.float32,
                initializer=tf.constant_initializer(.0))
            init_p = self.action_dim * [
                self.config["lbd"] * np.identity(input_obs.shape[1])
            ]
            init_p = np.stack(init_p)
            p = tf.get_variable(
                name="p",
                shape=(self.action_dim, input_obs.shape[1],
                       input_obs.shape[1]),
                dtype=tf.float32,
                initializer=tf.constant_initializer(init_p))
            # y_a=w_{a}^{T}x for each a \in A
            # (A, k) * (bs, 1, k) = (bs, A, k), then reduce into (bs, A)
            mean = tf.reduce_sum(w * tf.expand_dims(input_obs, 1), -1)
            # var_a = x^{T}P_{a}x for each a \in A
            # (|A|, K, K) * (batch_size, 1, 1, K) = (batch_size, |A|, K, K)
            # and reduce the last axis leading to (batch_size, |A|, K)
            # which could be viewed as P_{a}x for each a \in A
            P_multiplied_by_x = tf.reduce_sum(
                p * tf.reshape(input_obs, [-1, 1, 1, input_obs.shape[1]]), -1)
            # (batch_size, |A|, K) * (batch_size, 1, K) = (batch_size, |A|, K)
            # and reduce the last axis leading to (batch_size, |A|)
            var = tf.reduce_sum(
                P_multiplied_by_x * tf.expand_dims(input_obs, 1), -1)
            # standard deviation = \sqrt(variance)
            stddev = tf.sqrt(var)
        # for convenience of var reuse
        self.w = w
        self.p = p
        return mean, stddev

    def _build_loss(self, estimated_values, selected_arms, target_values):
        # (batch_size, |A|)
        mask = tf.one_hot(selected_arms, self.action_dim)
        # reduce the last axis of (batch_size, |A|) into (batch_size,)
        estimated = tf.reduce_sum(estimated_values * mask, 1)
        return tf.reduce_mean(0.5 * (estimated - target_values)**2)

    def _build_train(self,
                     input_obs,
                     estimated_values,
                     selected_arms,
                     target_values,
                     global_step=None):
        #w = tf.get_variable(name="w")
        #p = tf.get_variable(name="p")
        w = self.w
        p = self.p

        # select y_a for each (x, a) in the mini-batch
        # (bs, |A|)
        mask = tf.one_hot(selected_arms, self.action_dim)
        # reduce into (bs,)
        estimated = tf.reduce_sum(estimated_values * mask, 1)
        # (bs,) - (bs,)
        alpha = target_values - estimated

        # (|A|, K, K) * (bs, |A|, 1, 1) = (bs, |A|, K, K)
        p_a = p * tf.expand_dims(tf.expand_dims(mask, -1), -1)
        # (batch_size, K, K)
        p_a = tf.reduce_sum(p_a, 1)
        # p_a * x for each (x, a) in the mini-batch
        # (bs, K, K) * (bs, 1, K) = (bs, K, K) and reduce into (bs, K)
        pax = tf.reduce_sum(p_a * tf.expand_dims(input_obs, 1), -1)
        # compute x^{T}p_{a}x whose shape is (bs,, 1)
        # (bs, 1) * (bs, K) = (bs, K)
        g = 1.0 / (self.config["lbd"] + tf.reduce_sum(
            pax * input_obs, -1, keep_dims=True)) * pax

        # (bs, K, -1) * (bs, K, K) = (bs, K, K)
        gc = tf.expand_dims(input_obs, -1) * p_a
        # reduce into (bs, K)
        gc = tf.reduce_sum(gc, 1)

        with tf.control_dependencies([alpha, g]):
            # update w according to w = w + \alpha * g
            # \alpha * g has shape (bs, 1) * (bs, K) = (bs, K)
            delta_w = tf.expand_dims(alpha, -1) * g
            # (bs, 1, K) * (bs, |A|, 1) = (bs, |A|, K)
            delta_w = tf.expand_dims(delta_w, 1) * tf.expand_dims(mask, 2)
            # reduce into (|A|, K)
            delta_w = tf.reduce_sum(delta_w, 0)
            update_w_op = tf.assign_add(w, delta_w)
            # update p
            # (bs, K, 1) * (bs, 1, K) = (bs, K, K)
            delta_p = tf.expand_dims(g, -1) * tf.expand_dims(gc, 1)
            # (bs, 1, K, K) * (bs, |A|, 1, 1) = (bs, |A|, K, K)
            delta_p = delta_p * tf.expand_dims(tf.expand_dims(mask, -1), -1)
            # and reduce into (|A|, K, K)
            delta_p = tf.reduce_sum(delta_p, 0)
            update_p_op = tf.assign_add(p, -1.0 * delta_p)
        update_op = tf.group(*(update_w_op, update_p_op))
        with tf.control_dependencies([update_op]):
            update_op_with_step_cnt = tf.assign_add(global_step, 1)
        return update_op
