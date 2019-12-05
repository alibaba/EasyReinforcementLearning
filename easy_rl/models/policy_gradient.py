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
import tensorflow as tf

from easy_rl.models.model_base import Model
from easy_rl.utils import *


class PGModel(Model):
    """Vanilla Policy Gradient agent as described by [Sutton et al. (1999)]
    (https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)"""

    def __init__(self, obs_space, action_space, scope="pg_model", **kwargs):

        _default_config = {

            # network config
            # specify the parameters of network for each input state.
            # a default config will be used if empty
            # * it is recommended to overrider the `encode_obs` function and then implement a custom network structure.
            "network_spec": {},

            # training config
            # parameter for grad clipping
            "global_norm_clip": 40,

            # initialization of learning rate
            "init_lr": 0.001,

            # strategy of learning rate
            "lr_strategy_spec": {}
        }

        self.config = copy.deepcopy(_default_config)
        self.config.update(kwargs.get('model_config', {}))

        super(PGModel, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            scope=scope,
            **kwargs)

    @property
    def extra_act_fetches(self):
        """return the sample_action with state_value.
        """

        return {
            "value_preds": self.value_preds,
        }

    @property
    def extra_learn_fetches(self):
        """return the additional fetches when training
        """
        return {"loss": self.loss_op, "loss_v": self.loss_v}

    @property
    def learn_feed(self):
        """return the dict of placeholder of obs, actions, advantages and value_targets.
        the inputs will be used to optimize model with policy gradient
        """

        return {
            "obs": self.obs_ph,
            "actions": self.actions_ph,
            "advantages": self.advantages_ph,
            "value_targets": self.value_targets_ph
        }

    def _build_loss(self, value_preds, actions, advantages, value_targets):
        """implement specific loss function according to the optimization target

        Arguments:
            value_preds: predictions of state value from the policy network
            action: action taken by the behavior policy.
            advantages: state-action dependent advantage.
            value_targets: target for value function.
        """

        log_p = self.action_dist.log_p(action=actions)
        loss_op = -log_p * (advantages)
        self.summary_ops["train"].append(
            tf.summary.histogram("advantage", advantages))

        loss_v = tf.square(value_targets - value_preds)
        self.loss_v = tf.reduce_mean(loss_v)
        loss_op += 0.5 * loss_v
        self.summary_ops["train"].append(
            tf.summary.histogram("valu_targets", value_targets))
        self.summary_ops["train"].append(
            tf.summary.histogram("value_preds", value_preds))

        loss_op = tf.reduce_mean(loss_op)
        self.summary_ops["train"].append(tf.summary.scalar("loss", loss_op))

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

        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            self.preprocessed_obs_ph, self.preprocessed_next_obs_ph = self.preprocess_obs(
                self.obs_ph, self.next_obs_ph)
            # encode the input obs
            self.obs_embedding, self.value_preds = self._encode_obs(
                input_obs=self.preprocessed_obs_ph, scope="encode_obs")
            self.model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/encode_obs")
            # build output action_op
            self.action_op = self._actions_distribution(self.obs_embedding)

            # build loss_op
            self.loss_op = self._build_loss(
                actions=self.actions_ph,
                advantages=self.advantages_ph,
                value_preds=self.value_preds,
                value_targets=self.value_targets_ph)

        # get or create global_step
        self.global_step = tf.train.get_or_create_global_step()

        if kwargs.get("is_replica", False):
            return

        # build train_op
        init_lr = self.config.get('init_lr', 1e-3)
        lr_strategy_spec = self.config.get('lr_strategy_spec', {})

        # apply different decay strategy of learning rate
        lr = learning_rate_utils.LearningRateStrategy(
            init_lr=init_lr, strategy_spec=lr_strategy_spec)(self.global_step)
        self.summary_ops["train"].append(
            tf.summary.scalar("learning_rate", lr))

        opt = tf.train.AdamOptimizer(learning_rate=lr)

        self.train_op = self._build_train(
            loss=self.loss_op,
            optimizer=opt,
            vars=self.model_vars,
            global_step=self.global_step)

    def _encode_obs(self, input_obs, scope="encode_obs"):
        """build network to encode input feature.the desired output
        consists of two parts: action_logits and value_estimator

        Arguments:
          input_obs: the (list, dict)[of] input tensor of observation.
          scope: the name of variable scope.
        """

        with tf.variable_scope(name_or_scope=scope):
            # override the function `build_graph` to implement the specific network manually

            if len(self.config.get('network_spec')) == 0:
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

        super(PGModel, self)._build_ph_op(
            obs_space=obs_space, action_space=action_space)

        # add extra placeholder of advantages, value_targets for training
        self.advantages_ph = tf.placeholder(dtype=tf.float32, shape=[None])
        self.value_targets_ph = tf.placeholder(dtype=tf.float32, shape=[None])
