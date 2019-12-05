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

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import sys

from easy_rl.agents import AgentBase
from easy_rl import models
from easy_rl.models import PGModel
from easy_rl.utils.buffer import TrajectoryBuffer
from easy_rl.utils.utils import compute_targets


class A3CAgent(AgentBase):
    """A3C where workers interact with the environments and update the model parameters (on the servers) simutaneously without strict consistency considerations.

    See http://arxiv.org/abs/1602.01783 for more details.
    """

    _agent_name = "A3C"
    _valid_model_classes = [PGModel]
    _default_model_class_names = ["A3C"]

    def _init(self, model_config, ckpt_dir, custom_model):

        model_config = model_config or {}

        if custom_model is not None:
            assert np.any([
                issubclass(custom_model, e) for e in self._valid_model_classes
            ])
            model_class = custom_model
        else:
            model_name = model_config.get("type", self._default_model_class)
            assert model_name in self._valid_model_class_names, "{} does NOT support {} model.".format(
                self._agent_name, model_name)
            model_class = models.models[model_name]
        with self.distributed_handler.device:
            self.model = model_class(
                self.executor.ob_ph_spec,
                self.executor.action_ph_spec,
                model_config=model_config)
            self._behavior_model = self.model

            self._global_num_sampled = tf.get_variable(
                name="global_num_sampled", dtype=tf.int64, shape=())

        self._update_global_num_sampled = tf.assign_add(
            self._global_num_sampled,
            np.int64(self.config["sample_batch_size"]),
            use_locking=True)

        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        self.executor.setup(
            self.distributed_handler.master,
            self.distributed_handler.is_chief,
            self.model.global_step,
            ckpt_dir,
            self.model.summary_ops,
            global_vars=global_vars,
            local_vars=None)

        # transitions are stored in chronological order
        self._buffer = TrajectoryBuffer()

    def send_experience(self, obs, actions, rewards, dones, value_preds):
        """Store experience in the buffer

        Postprocess the collected transitions and store the them in the buffer.
        `obs`, `actions`, `rewards`, `next_obs`, `dones` are basic fields.

        Arguments:
            obs (list): the observed states.
            actions (list): the actions taken in response to the states.
            rewards (list): collected from the environment.
            next_obs (list): the obsered next states.
            dones (list): bool flags indicating whether the episode finishes.
            weights is the initializd priorities of samples.
            kwargs (dict) contains the additional algorithm-specific field(s).
        """

        # there is no need to compute_targets for v-trace model
        advantages, value_targets = compute_targets(
            rewards,
            dones,
            value_preds,
            gamma=self.config.get('gamma', 0.95),
            lambda_=self.config.get('lambda_', 1.0),
            use_gae=self.config.get('use_gae', True))
        traj_len = len(advantages)

        kwargs_actually_used = dict()
        kwargs_actually_used["advantages"] = advantages
        kwargs_actually_used["value_targets"] = value_targets
        self._buffer.add(
            obs=np.asarray(obs)[:traj_len],
            actions=np.asarray(actions)[:traj_len],
            rewards=np.asarray(rewards)[:traj_len],
            dones=np.asarray(dones)[:traj_len],
            **dict((k, v) for k, v in kwargs_actually_used.items()))

        if len(self._buffer) >= self.config["batch_size"]:
            self._ready_to_receive = True

        self._ready_to_send = False

        self.executor.run(self._update_global_num_sampled, {})

        # clear the lists
        del obs[:len(obs) - 1]
        del actions[:len(actions) - 1]
        del rewards[:len(rewards) - 1]
        del dones[:len(dones) - 1]
        del value_preds[:len(value_preds) - 1]

    def receive_experience(self):
        """Acquire training examples from the buffer.

        Returns:
            batch_data (dict): contains fields of data for making an update.
        """

        self._ready_to_receive = False
        return self._buffer.sample(self.config["batch_size"])

    def _init_act_count(self):

        # In order to get the same size of value_preds of next_obs, one more timestep is needed in the first iteration.
        self._act_count = -1

    def join(self):
        """Call `server.join()` if the agent object serves as a parameter server.
        """

        self.distributed_handler.server.join()

    def should_stop(self):
        """Judge whether the agent should stop working.
        """
        if self.distributed_handler.job_name == "worker":
            global_timesteps = self.executor.run(self._global_num_sampled, {})
            if global_timesteps > self.config.get("scheduled_timesteps",
                                                  1000000):
                return True
        return False
