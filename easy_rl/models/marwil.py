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
from easy_rl.utils import *


class MarwilModel(Model):
    """implemention of Exponentially Weighted Imitation Learning for
    Batched Historical Data[Qing Wang et al., 2018](https://papers.nips.cc/paper/
    7866-exponentially-weighted-imitation-learning-for-batched-historical-data.pdf)
    """

    def __init__(self, obs_space, action_space, scope="marwil_model",
                 **kwargs):

        _default_config = {

            # network config
            # specify the parameters of network for each input state.
            # a default config will be used if empty
            # * it is recommended to overrider the `encode_states` function and then implement a custom network structure.
            "network_spec": {},

            # model config
            # coefficient for emphasis on policy improvement
            # set to 0 and the algorithm degenerates to ordinary imitation learning
            "beta": 1.0,

            # normalization factor for advantage
            "norm_c": 1.0,

            # coefficient for moving average of normalization factor
            "gamma": 1e-8,

            # training config
            # coefficient for loss of value function
            "vf_coef": 0.5,

            # parameter for grad clipping
            "global_norm_clip": 40,

            # initialization of learning rate
            "lr_init": 0.001,
        }

        self.config = copy.deepcopy(_default_config)
        self.config.update(kwargs.get('model_config', {}))

        super(MarwilModel, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            scope=scope,
            **kwargs)

    @property
    def extra_act_fetches(self):
        """return the sample_action with action logits and state_value.
        """

        return {"value_preds": self.value_preds}

    @property
    def learn_feed(self):
        """return the dict of placeholder for training.
        """

        return {
            "obs": self.obs_ph,
            "actions": self.actions_ph,
            "value_target": self.value_target_ph
        }

    def _build_loss(self, value_preds, actions, value_target):
        """implement specific loss function according to the optimization target

        Arguments:
            value_preds: estimator of state value from the behavior policy
            actions: action taken by the behavior policy.
            value_target: target for value function.
        """

        # value loss
        loss_v = .5 * tf.reduce_mean(tf.nn.l2_loss(value_target - value_preds))

        log_target_prob = self.action_dist.log_p(action=actions)
        beta = self.config.get('beta', 1.0)

        advantage = tf.stop_gradient(
            (value_target - value_preds) / self.norm_c)

        # update norm_c with moving average
        gamma = self.config.get('gramma', 1e-8)
        update_c = self.norm_c + gamma * tf.reduce_mean(advantage)

        # maxminize E ~ exp(beta * A(s,a))log(pi(s,a))
        with tf.control_dependencies([update_c]):
            loss_p = -tf.reduce_mean(
                tf.exp(beta * advantage) * log_target_prob)

        vf_coef = self.config.get('vf_coef', 0.5)

        # total loss
        loss_op = loss_p + vf_coef * loss_v

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

            # encode the input obs
            self.obs_embedding, self.value_preds = self._encode_obs(
                input_obs=self.obs_ph, scope="encode_obs")
            self.model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/encode_obs")
            self.norm_c = tf.get_variable(
                name="normalization_coefficient",
                dtype=tf.float32,
                initializer=self.config.get('norm_c', 1.0))

            # build output action_op
            self.action_op = self._actions_distribution(self.obs_embedding)

            # build loss_op
            self.loss_op = self._build_loss(
                value_preds=self.value_preds,
                actions=self.actions_ph,
                value_target=self.value_target_ph)

        # get or create global_step
        self.global_step = tf.train.get_or_create_global_step()

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
            loss=self.loss_op, optimizer=opt, global_step=self.global_step)

    def _encode_obs(self, input_obs, scope="encode_obs"):
        """build network to encode input feature.the desired output
        consists of two parts: action_logits and value_estimator

        Arguments:
          input_obs: the (list, dict)[of] input tensor of state.
          scope: the name of variable scope
        """

        with tf.variable_scope(name_or_scope=scope):
            # override the function `build_graph` to implement the specific network manually

            if len(self.config.get('network_spec')) == 0:
                # must set action_dim to use default network
                action_dim = self.config.get('action_dim')

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

    def _build_ph_op(self, obs_space, action_space):

        super(MarwilModel, self)._build_ph_op(
            obs_space=obs_space, action_space=action_space)

        # add extra placeholder of return for training
        self.value_target_ph = tf.placeholder(dtype=tf.float32, shape=[None])
