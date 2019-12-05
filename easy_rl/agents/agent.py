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

import easy_rl.models as models
from easy_rl.models import DQNModel, DDPGModel, PGModel, PPOModel, VTraceModel, EvolutionStrategy, LinUCBModel
from easy_rl.agents import AgentBase
from easy_rl.utils.buffer import ReplayBuffer, PrioritizedReplayBuffer, TrajectoryBuffer
from easy_rl.utils.utils import n_step_adjustment, compute_targets


class Agent(AgentBase):
    """(single-machine) Agent

    Different behaviors (in RL jargon) can be made by specifying different kinds of model.

    Attributes:
        executor (obj): Handle the runtime.
        model (obj): Provide the computation graph.
        distributed_handler (obj): Provide the context for distributed computing.
        _buffer (obj): Store the collected experience.
    """

    _agent_name = "Agent"
    _default_model_class = "DQN"
    _valid_model_classes = [
        DQNModel, DDPGModel, PGModel, PPOModel, VTraceModel, EvolutionStrategy,
        LinUCBModel
    ]
    _valid_model_class_names = [
        "DQN", "DDPG", "PG", "PPO", "ES", "Vtrace", "LinUCB"
    ]

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

        self.executor.setup(
            self.distributed_handler.master,
            self.distributed_handler.is_chief,
            self.model.global_step,
            ckpt_dir,
            self.model.summary_ops,
            save_steps=10)

        # Under single machine setting, we create buffer object as the class attribute
        # The type of buffer should be determined by the model type
        if issubclass(model_class, DQNModel) or issubclass(
                model_class, DDPGModel):
            if self.config.get("prioritized_replay", False):
                self._buffer = PrioritizedReplayBuffer(
                    self.config["buffer_size"],
                    self.config["prioritized_replay_alpha"])
            else:
                self._buffer = ReplayBuffer(self.config["buffer_size"])
        else:
            # transitions are stored in chronological order
            self._buffer = TrajectoryBuffer()

    def send_experience(self,
                        obs,
                        actions,
                        rewards,
                        dones,
                        next_obs=None,
                        weights=None,
                        **kwargs):
        """Store experience in the buffer.

        Postprocess the collected transitions and store them in the buffer.
        `obs`, `actions`, `rewards`, `next_obs`, `dones` are basic fields.

        Arguments:
            obs (list): the observed states.
            actions (list): the actions taken in response to the states.
            rewards (list): collected from the environment.
            next_obs (list): the obsered next states.
            dones (list): bool flags indicating whether the episode finishes.
            weights (list): the initializd priorities of samples.
            kwargs (dict): contains the additional algorithm-specific field(s).
        """

        n_step = self._model_config.get("n_step", 1)

        if isinstance(self._buffer, TrajectoryBuffer):
            # there is no need to compute_targets for v-trace model
            if self.config.get("compute_targets", True):
                advantages, value_targets = compute_targets(
                    rewards,
                    dones,
                    kwargs.get("value_preds", None),
                    gamma=self.config.get('gamma', 0.95),
                    lambda_=self.config.get('lambda_', 1.0),
                    use_gae=self.config.get('use_gae', True))
                traj_len = len(advantages)
                kwargs_actually_used = dict(
                    (k, np.asarray(v)[:traj_len]) for k, v in kwargs.items())
                kwargs_actually_used["advantages"] = advantages
                kwargs_actually_used["value_targets"] = value_targets
                self._buffer.add(
                    np.asarray(obs)[:traj_len],
                    np.asarray(actions)[:traj_len],
                    np.asarray(rewards)[:traj_len],
                    np.asarray(dones)[:traj_len],
                    **dict((k, v) for k, v in kwargs_actually_used.items()))
            else:
                kwargs_actually_used = kwargs
                self._buffer.add(
                    np.asarray(obs), np.asarray(actions), np.asarray(rewards),
                    np.asarray(dones),
                    **dict((k, np.asarray(v))
                           for k, v in kwargs_actually_used.items()))

            if len(self._buffer) >= self.config["batch_size"]:
                self._ready_to_receive = True
        else:
            _obs, _actions, _rewards, _next_obs, _dones = n_step_adjustment(
                obs, actions, rewards, next_obs, dones,
                self._model_config.get("gamma", 0.99), n_step)
            self._buffer.add(
                obs=_obs,
                actions=_actions,
                rewards=_rewards,
                dones=_dones,
                next_obs=_next_obs,
                weights=weights,
                **{})
            if len(self._buffer) > self.config.get("learning_starts", 0):
                self._ready_to_receive = True
        self._ready_to_send = False

        # clear the lists
        n_step += 1 if (isinstance(self.model, PPOModel)
                        or isinstance(self.model, PGModel)) else 0
        del obs[:len(obs) - n_step + 1]
        del actions[:len(actions) - n_step + 1]
        del rewards[:len(rewards) - n_step + 1]
        del next_obs[:len(next_obs) - n_step + 1]
        del dones[:len(dones) - n_step + 1]
        for k in kwargs.keys():
            del kwargs[k][:len(kwargs[k]) - n_step + 1]

    def receive_experience(self):
        """Acquire training examples from the buffer.

        Returns:
            batch_data (dict): contains fields of data for making an update.
        """

        self._ready_to_receive = False
        if isinstance(self._buffer, TrajectoryBuffer):
            batch_data = self._buffer.sample(self.config["batch_size"])
        elif isinstance(self._buffer, PrioritizedReplayBuffer):
            batch_data = self._buffer.sample(
                self.config["batch_size"],
                beta=self.config["prioritized_replay_beta"])
        else:
            batch_data = self._buffer.sample(self.config["batch_size"])
            batch_data["weights"] = np.ones(
                batch_data["rewards"].shape, dtype=np.float32)
        return batch_data

    def update_priorities(self, indexes, td_error):
        """ Update the priorities of the sampled data.

        Arguments:
            td_error (np.ndarray): the computed TD errors.
        """
        if isinstance(self._buffer, PrioritizedReplayBuffer):
            self._buffer.update_priorities(
                indexes=indexes, priorities=(np.abs(td_error) + 1e-6))

    def _init_act_count(self):

        if (isinstance(self.model, PPOModel) or isinstance(
                self.model, PGModel)) and not hasattr(self, "_act_count"):
            # In order to get the value_preds of next_obs, one more timestep is needed in the first iteration.
            self._act_count = -1
        elif (isinstance(self.model, DQNModel) or isinstance(
                self.model, DDPGModel)) and not hasattr(self, "_act_count"):
            self._act_count = -self._model_config.get("n_step", 1) + 1
        else:
            self._act_count = 0
