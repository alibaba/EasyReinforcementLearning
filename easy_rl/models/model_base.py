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

from collections import OrderedDict
import tensorflow as tf

from easy_rl.utils import *


class Model(object):
    """The base model defines the main interface including inference and training.
    Models that implements a specific algorithm should be derived from the base model.

    Attributes:
        output_actions (tensor): the output action for inference.
        extra_act_fetches (list): the extra output that need to be fetched along with output_actions.
        extra_act_feed (dict): the extra input that need to be fed when inference.
        loss (tensor): the result of loss function.
        update_op (tensor): the operator to trigger optimization.
        extra_learn_fetches (list): the extra output that need to be fetched along with the optimization.
        extra_learn_feed (dict): the extra input that need to be fed when training.
    """

    def __init__(self, obs_space, action_space, scope="model", **kwargs):
        self.summary_ops = dict(train=[], act=[], extra=[])
        self._build_ph_op(obs_space=obs_space, action_space=action_space)
        self._build_graph(scope=scope, **kwargs)
        self.add_extra_summary_op()

    @property
    def output_actions(self):
        """return the output actions"""
        return self.action_op

    @property
    def extra_act_fetches(self):
        """return the extra fetches that needs to be run with `action_op`
        """

        return {}

    @property
    def extra_act_feed(self):
        """return the extra fetches that needs to be run with `action_op`
        """

        return {}

    @property
    def loss(self):
        """return the loss"""

        return self.loss_op

    @property
    def update_op(self):
        """return the train_op to optimize params of the model"""

        return self.train_op

    @property
    def extra_learn_fetches(self):
        """return the extra fetches that needs to be run with `train_op`"""

        return {}

    @property
    def learn_feed(self):
        """return the placeholder (obs,action,reward,terminal,next_obs) that
        needs to be fed when calling the `train_op`
        """

        return {}

    @property
    def variables(self):
        """return all the variables
        """

        return []

    def preprocess_obs(self, obs, next_obs=None):
        """Describe computation graph for preprocessing observations
        where bandit models do not consider next observation.
        Arguments:
            obs (obj): Tensor of current observation
            next_obs (obj): Tensor of next observation
        """
        # currently, this is a dummy implementation.
        return obs, next_obs

    def _build_graph(self, scope, **kwargs):
        """Construct the whole model graph with input placeholders.
        the required attributes including action_op, train_op, and other
        necessary ops.

        Arguments:
            scope: the name of variable scope
            kwargs: extra params used to build the graph

        Examples:
            In the context of reinforcement learning, there are mainly three
            kinds of algorithms: value-based, policy gradients, and
            actor-critic. We present their skeletons respectively. Overall,
            EasyRL poses as least constraints as possible over users
            (i.e., developers), so that they can describe their
            algorithms in this function (as well as others called in this
            function) with most freedom, say that naming the intermediate TF
            tensors arbitrarily.
             
            (jones.wz) TO DO: provide the code snippets here.
            Value-based:
            Policy gradients:
            Actor-critic:
                pi_logits, state_value = self._encode_obs(obs_dict=self.obs_ph_dict,
                                                       scope="encode_obs")
                self.action_op = self._actions_distribution(pi_logits)

                self.R = utils.discounted_cumulative_reward(terminal=self.terminal_ph_dict,
                                                           reward=self.reward_ph_dict,
                                                           discount=0.99,
                                                           final_reward=state_value[-1],
                                                           horizon=0)

                self.loss_op = self._build_loss(logits=pi_logits,
                                                actions=self.action_pl_dict,
                                                R=self.R,
                                                value=state_value[-1])

                opt = tf.train.AdamOptimizer()

                global_step = tf.train.get_or_create_global_step()

                self.train_op = self._build_train(loss=self.loss_op,
                                                  optimizer=opt,
                                                  global_step=global_step)
        """
        raise NotImplementedError

    def _encode_obs(self, input_obs, scope="encode_obs"):
        """build network here and the output is a presentation of encoded
        obs. It`s the main place to implement function approximator with
        neural network.

        Arguments:
          input_obs: the (list, dict)[of] input tensor of observation.
          scope: the name of variable scope

        Returns:
            the encode of input feature.

        Examples:
            input = obs_dict["ob"]
            with tf.variable_scope(name_or_space=scope, reuse=tf.AUTO_REUSE):
                obs_embedding = tf.layers.dense(input, 3, tf.nn.relu)
                return obs_embedding
        """
        raise NotImplementedError

    def _actions_distribution(self, obs_embedding, **kwargs):
        """choose different distribution functions according to the `action_distribution_type`
        both deterministic and stochastic sampling strategy is needed. sample actions according
        to the parameter `obs_embedding` and `deterministic`

        Arguments:
            obs_embedding : one (or nested) vector presentation of observation.
            kwargs: extra parameters for exploration.

        Returns:
            output_action for inference.
        """

        # set `parameter_noise` to True and the epsilon-exploration will be forbidden
        params_noise = self.config.get("parameter_noise", False)
        self.action_dist = action_utils.get_action_distribution(
            obs_embedding, self.action_distribution_type,
            self.deterministic_ph, self.eps_ph
            if (hasattr(self, 'eps_ph')
                and not params_noise) else None, **kwargs)

        return self.action_dist.get_action()

    def _build_ph_op(self, obs_space, action_space):
        """initialize all the required placeholders for building model graph, including placeholders for inference and training

        Arguments:
            obs_space: the definition of observation space. nested structure is forbidden.
                          Supported formats include list, dict, and singleton.
            action_space: the definition of output action. Currently only single outputs is supported

        Examples:
            obs_space = [{"type":tf.int32, "shape":[None, 10]}, {"type":tf.float32, "shape":[None, 20]}]
            action_space = {"type": "Categorical", "action_dim" : 5}
        """

        if isinstance(obs_space, list):
            self.obs_ph = [
                tf.placeholder(
                    dtype=v[0], shape=v[1], name="the_{}_obs".format(i))
                for i, v in enumerate(obs_space)
            ]
            self.next_obs_ph = [
                tf.placeholder(
                    dtype=ph.dtype,
                    shape=ph.shape,
                    name='the_{}_next_obs'.format(i))
                for i, ph in enumerate(self.obs_ph)
            ]
        elif isinstance(obs_space, dict):
            self.obs_ph = {
                name: tf.placeholder(
                    dtype=v[0], shape=v[1], name="{}_obs".format(name))
                for name, v in obs_space.items()
            }
            self.next_obs_ph = {
                'next_' + name: tf.placeholder(
                    dtype=v[0], shape=v[1], name="next_{}_obs".format(name))
                for name, v in obs_space.items()
            }
        else:
            self.obs_ph = tf.placeholder(
                dtype=obs_space[0], shape=obs_space[1], name="obs")
            self.next_obs_ph = tf.placeholder(
                dtype=obs_space[0], shape=obs_space[1], name="next_obs")

        # get action_distribution_type, the available value is one of (`Categorical`, `DiagGaussian`, `Identity`)
        self.action_dim = action_space[0]
        self.action_distribution_type = self.config.get(
            'action_distribution_type', {}) or action_space[1]

        if self.action_distribution_type == "Categorical":
            self.actions_ph = tf.placeholder(
                dtype=tf.int32, shape=[None], name="actions")
        else:
            self.actions_ph = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.action_dim],
                name="actions")

        self.rewards_ph = tf.placeholder(
            dtype=tf.float32, shape=[None], name="rewards")
        self.dones_ph = tf.placeholder(
            dtype=tf.float32, shape=[None], name="dones")
        self.deterministic_ph = tf.placeholder(
            dtype=tf.bool, shape=[], name="deterministic")
        # create placeholder for indicate training stage
        self.is_training_ph = tf.placeholder_with_default(
            True, shape=(), name="is_training")
        # dummy ph to set summary flag in runtime
        self._set_summary_flag = tf.placeholder_with_default(
            "train", shape=(), name="set_summary_flag")

    def _build_loss(self, **kwargs):
        """implement specific loss function according to the optimization target

        Arguments:
            kwargs : params used to build the loss function.

        Returns:
            result of the loss function.

        Examples:
            logit = kwargs.get('logit')
            action = kwargs.get('action')
            R = kwargs.get('R')
            value = kwargs.get('value')

            log_p = tf.reduce_sum(tf.nn.log_softmax(logits) * tf.one_hot(actions, ), axis=-1)

            loss_op = - log_p * (R - value)

            return loss_op
        """

        raise NotImplementedError

    def _build_train(self, **kwargs):
        """add optimizer here and minimize the loss.
        different learning rate strategies and gradients post-process can be applied here

        Arguments:
            kwargs : params used to build the training operation.

        Returns:
            operation for training.

        Examples:
            loss = kwargs.get('loss')
            lr = kwargs.get("input_lr", 0.001)

            opt = kwargs.get('optimizer')
            vars = kwargs.get('vars', None)
            global_step = kwargs.get('global_step', None)
            if global_step:
                lr = tf.train.exponential_decay(lr, global_step, 20000, decay_rate=0.9)
            train_op = opt.minimize(loss, global_step=global_step, var_list=vars)
        """

        raise NotImplementedError

    def add_extra_summary_op(self):
        """add extra summary op.

        summary_ops add in this function will be exported when the `session.run` called in training stage.
        Note: be careful to add summary_op, any input of summary_op missing will raise an error.
        """
        self.extra_episode_return = tf.placeholder_with_default(
            tf.constant([0.0], dtype=tf.float32),
            shape=[None],
            name="episode_return")
        self.summary_ops["extra"].extend([
            tf.summary.scalar(
                name="episode_return",
                tensor=tf.reduce_mean(self.extra_episode_return))
        ])
