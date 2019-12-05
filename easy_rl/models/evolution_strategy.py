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


class EvolutionStrategy(Model):
    def __init__(self, obs_space, action_space, scope="es_model", **kwargs):

        _default_config = {

            # network config
            # specify the parameters of network for each input state.
            # a default config will be used if empty
            # * it is recommended to overrider the `encode_obs` function and then implement a custom network structure.
            "network_spec": {},

            # size of noise table
            "noise_table_size": 25000000,

            # seed to initialize the noise table
            # Note that the seed must be same in different worker to ensure
            "global_seed": 0,

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

        super(EvolutionStrategy, self).__init__(
            obs_space=obs_space,
            action_space=action_space,
            scope=scope,
            **kwargs)

    @property
    def perturbed_actions(self):
        return self.perturbed_actions_op

    @property
    def learn_feed(self):
        """return the dict of placeholder for training.
        the input value will be used to optimize model.
        """

        return {
            "perturbation_seeds": self.perturbation_seeds_ph,
            "returns": self.R_ph,
            "perturbation_scales": self.perturbation_scales_ph
        }

    @property
    def extra_learn_fetches(self):
        """return the extra fetches when training.
        """

        return {}

    @property
    def perturbation_feed(self):
        """return the placeholder of perturbation seed
        """

        return {
            "perturbation_seeds": self.perturbation_seeds_ph,
            "perturbation_scales": self.perturbation_scales_ph,
            "positive_perturbation": self.positive_perturbation_ph
        }

    @property
    def all_variables(self):
        """return all variables of model object"""
        return self._scope_vars

    @property
    def actor_sync_variables(self):

        return self.model_vars

    def reset_perturbed_parameters(self):
        """generate a candidate perturbation and reset the perturbed model with latest """

        params_size = sum(
            [utils.prod(v.shape.as_list()) for v in self.model_vars])

        perturbations = self._generate_perturbations(
            params_size=params_size, seed=self.perturbation_seeds_ph)
        perturbations = tf.squeeze(perturbations)

        var_shapes = [v.shape.as_list() for v in self.model_vars]
        perturbations_list = utils.unflatten_tensor(perturbations, var_shapes)

        ops = []
        for var, perturbed_var, noise in zip(self.model_vars,
                                             self.perturbed_model_vars,
                                             perturbations_list):
            ops.append(
                tf.assign(
                    perturbed_var, var +
                    (2 * tf.cast(self.positive_perturbation_ph, tf.float32) -
                     1.0) * self.perturbation_scales_ph * noise))

        return tf.group(*ops)

    def _build_train(self, seed, R, optimizer, vars, global_step=None):

        # compute the gradient
        # here we want to maximize the reward
        centered_R = self._get_centered_ranks_tensor(-1.0 * R)
        positive_R, negative_R = tf.split(centered_R, 2, axis=1)
        weights = tf.reshape((positive_R - negative_R), [1, -1])

        params_size = sum([utils.prod(v.shape.as_list()) for v in vars])
        noise_size = tf.shape(seed)[0]
        perturbations = self._generate_perturbations(
            params_size=params_size, seed=self.perturbation_seeds_ph)
        perturbations = perturbations * tf.expand_dims(
            self.perturbation_scales_ph, axis=-1)
        delta = tf.matmul(weights, perturbations)

        # add summary
        self.summary_ops["train"].append(
            tf.summary.scalar("perturbation_scales",
                              tf.reduce_mean(self.perturbation_scales_ph)))
        self.summary_ops["train"].append(
            tf.summary.histogram("perturbations", perturbations))
        self.summary_ops["train"].append(
            tf.summary.scalar("mean_noise_episode_returns", tf.reduce_mean(R)))
        self.summary_ops["train"].append(
            tf.summary.scalar("max_noise_episode_returns", tf.reduce_max(R)))
        self.summary_ops["train"].append(
            tf.summary.scalar("min_noise_episode_returns", tf.reduce_min(R)))

        delta = tf.reshape(delta,
                           (-1, )) / tf.cast(noise_size, tf.float32) / 2.0

        var_shapes = [v.shape.as_list() for v in vars]
        gradients = utils.unflatten_tensor(delta, var_shapes)

        # apply grad clipping
        with tf.control_dependencies(gradients):
            clipped_grads, _ = tf.clip_by_global_norm(
                gradients, clip_norm=self.config.get('global_norm_clip', 40))
            grads_and_vars = list(zip(clipped_grads, vars))

            train_op = optimizer.apply_gradients(
                grads_and_vars, global_step=global_step)

        return train_op

    def _build_graph(self, scope, **kwargs):

        # get or create global_step
        self.global_step = tf.train.get_or_create_global_step()
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):

            # create noise table to generate random perturbation
            noise_table_size = self.config.get('noise_table_size', 25000000)
            global_seed = self.config.get('global_seed', 0)
            self.noise_table = tf.get_variable(
                name='noise_table',
                shape=(noise_table_size, ),
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(
                    mean=0.0, stddev=1.0, seed=global_seed, dtype=tf.float32),
                trainable=False)

            self.preprocessed_obs_ph, self.preprocessed_next_obs_ph = self.preprocess_obs(
                self.obs_ph, self.next_obs_ph)
            # encode the input obs
            self.obs_embedding = self._encode_obs(
                input_obs=self.preprocessed_obs_ph, scope="encode_obs")
            self.model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/encode_obs")

            # encode the input obs
            self.perturbed_obs_embedding = self._encode_obs(
                input_obs=self.preprocessed_obs_ph,
                scope="perturbed_encode_obs")
            self.perturbed_model_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES,
                scope=tf.get_variable_scope().name + "/perturbed_encode_obs")

            # build output action_op
            self.action_op = self._actions_distribution(self.obs_embedding)

            self.perturbed_actions_op = self._actions_distribution(
                self.perturbed_obs_embedding)

            # no need to create loss function with random search algorithm
            self.loss_op = tf.no_op()

            # reset perturbation
            self.reset_perturbation_op = self.reset_perturbed_parameters()

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
            self.opt = tf.train.AdamOptimizer(learning_rate=lr)

            self.train_op = self._build_train(
                self.perturbation_seeds_ph,
                self.R_ph,
                optimizer=self.opt,
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

    def _build_ph_op(self, obs_space, action_space):

        super(EvolutionStrategy, self)._build_ph_op(
            obs_space=obs_space, action_space=action_space)

        # add extra placeholder of advantage, return and logits for training
        self.perturbation_seeds_ph = tf.placeholder(
            dtype=tf.int32, shape=[None])
        self.positive_perturbation_ph = tf.placeholder(dtype=tf.bool, shape=())
        self.perturbation_scales_ph = tf.placeholder(
            dtype=tf.float32, shape=(None, ))
        self.R_ph = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    def _get_centered_ranks_tensor(self, R):
        flatten_R = tf.reshape(R, (-1, ))
        _, sorted_index = tf.nn.top_k(
            -1.0 * flatten_R, k=tf.shape(flatten_R)[0], sorted=False)
        perm_index = tf.cast(tf.invert_permutation(sorted_index), tf.float32)
        perm_index = tf.reshape(perm_index, tf.shape(R))
        centered_rank_R = perm_index / (
            tf.cast(tf.shape(flatten_R)[0], tf.float32) - 1.0) - 0.5

        return centered_rank_R

    def _generate_perturbations(self, params_size, seed):
        """generate the perturbation from fixed noise table
        Note that  `seed` is used as a random index"""

        seed = tf.mod(seed, tf.shape(self.noise_table)[0] - params_size)
        noise = tf.map_fn(
            lambda t: self.noise_table[t:t + params_size],
            seed,
            dtype=tf.float32)
        noise = tf.reshape(noise, [-1, params_size])
        return noise
