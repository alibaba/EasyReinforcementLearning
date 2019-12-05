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

import logging
import time
import six.moves.queue as Queue
import numpy as np

from easy_rl.agents import AsyncAgent
from easy_rl.models import VTraceModel
from easy_rl.utils.buffer import TrajectoryBuffer

logger = logging.getLogger(__name__)


class ImpalaAgent(AsyncAgent):
    """Impala, an async actor-learner architecture with V-trace to remedy any policy lag. see http://arxiv.org/abs/1802.01561 for details.
    """

    _agent_name = "Impala"
    _default_model_class = "Vtrace"
    _valid_model_classes = [VTraceModel]
    _valid_model_class_names = ["Vtrace"]

    def _create_buffer(self):
        return TrajectoryBuffer(self.config.get("buffer_size", 10000))

    def send_experience(self, obs, actions, rewards, dones, **kwargs):
        is_vec_env = kwargs.pop("vec_env", False)
        num_env = kwargs.pop("num_env", 1)
        if is_vec_env:
            # unstack batch data
            dones[-1] = [True] * num_env
            obs_ = np.swapaxes(np.asarray(obs), 0, 1)
            obs_ = np.reshape(obs_, (-1, ) + np.shape(obs_)[2:])
            dones_ = np.swapaxes(np.asarray(dones), 0, 1)
            dones_ = np.reshape(dones_, (-1, ) + np.shape(dones_)[2:])
            rewards_ = np.swapaxes(np.asarray(rewards), 0, 1)
            rewards_ = np.reshape(rewards_, (-1, ) + np.shape(rewards_)[2:])
            actions_ = np.swapaxes(np.asarray(actions), 0, 1)
            actions_ = np.reshape(actions_, (-1, ) + np.shape(actions_)[2:])

            kwargs_ = {}
            for k, v in kwargs.items():
                v_ = np.swapaxes(np.asarray(v), 0, 1)
                v_ = np.reshape(v_, (-1, ) + np.shape(v_)[2:])
                kwargs_[k] = v_
        else:
            obs_, actions_, rewards_, dones_ = obs, actions, rewards, dones
            kwargs_ = kwargs

        try:
            self._actor2mem_q.put([
                arr[:(self.config["sample_batch_size"] * num_env)]
                for arr in ([
                    np.asarray(obs_),
                    np.asarray(actions_),
                    np.asarray(rewards_),
                    np.asarray(dones_)
                ] + [np.asarray(v) for k, v in kwargs_.items()])
            ])
        except Queue.Full as e:
            logger.warn("{}".format(e))
        finally:
            pass
        self._ready_to_send = False

        # clear the lists
        del obs[:len(obs)]
        del actions[:len(actions)]
        del rewards[:len(rewards)]
        del dones[:len(dones)]
        for k in kwargs.keys():
            del kwargs[k][:len(kwargs[k])]

    def communicate(self):
        """Run this method on memory hosts

        Receive transitions from actors and add the data to replay buffers.
        Sample from the replay buffers and send the samples to learners.
        """

        if not self._actor2mem_q.empty():
            samples = self._actor2mem_q.get()
            obs, actions, rewards, dones, logits = samples
            self._buffer.add(obs, actions, rewards, dones, logits=logits)
            self._act_count += np.shape(rewards)[0]
            self._receive_count += np.shape(rewards)[0]
            if int(self._receive_count / 10000) > self._last_receive_record:
                self._last_receive_record = int(self._receive_count / 10000)
                self.executor.run(self._update_num_sampled_timesteps, {})

            if self._act_count >= self.config["batch_size"] and not self._mem2learner_q.full(
            ):
                batch_data = self._buffer.sample(self.config["batch_size"])
                obs, actions, rewards, dones, logits = batch_data["obs"], \
                    batch_data["actions"], batch_data["rewards"], batch_data["dones"], batch_data["logits"]
                # Note: the order of the elements should be the same as the order of OrderedDict from `learn_feed`.
                self._mem2learner_q.put([obs, actions, rewards, dones, logits])
                self._act_count = 0

    def receive_experience(self):
        if self._receive_q.empty():
            return None
        samples = self._receive_q.get()
        buffer_index, obs, actions, rewards, dones, logits = samples
        return dict(
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            logits=logits)
