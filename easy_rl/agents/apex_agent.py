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

import sys
import threading
import logging
import multiprocessing as mp
import six.moves.queue as Queue
import time
import numpy as np
import tensorflow as tf

from easy_rl.agents import AsyncAgent
from easy_rl.utils.buffer import ReplayBuffer, PrioritizedReplayBuffer, TrajectoryBuffer
from easy_rl.utils.utils import n_step_adjustment, compute_targets
from easy_rl.agents.monitor import ProxyReceiver, ProxySender
from easy_rl.models import DQNModel, DDPGModel

logger = logging.getLogger(__name__)


class ApexAgent(AsyncAgent):
    """Apex, an async actor-learner architecture. see http://arxiv.org/abs/1803.00933 for details.
    """

    _agent_name = "Apex"
    _default_model_class = "DQN"
    _valid_model_classes = [DQNModel, DDPGModel]
    _valid_model_class_names = ["DQN", "DDPG"]

    def _get_out_queue_meta(self):
        dtypes, shapes, phs = super(ApexAgent, self)._get_out_queue_meta()
        if self.config.get("prioritized_replay", False):
            ph = tf.placeholder(
                dtype=tf.int32, shape=(self.config["batch_size"], ))
            dtypes.append(ph.dtype)
            shapes.append(ph.shape)
            phs.append(ph)
        return dtypes, shapes, phs

    def _build_communication(self, job_name, task_index):
        super(ApexAgent, self)._build_communication(
            job_name=job_name, task_index=task_index)
        # DQN and DDPG need to update priorities
        self.update_phs = [
            tf.placeholder(
                dtype=tf.int32, shape=(self.config["batch_size"], )),
            tf.placeholder(
                dtype=tf.float32, shape=(self.config["batch_size"], ))
        ]
        if job_name in ["memory", "learner"]:
            self._update_queues = list()
            self._en_update_queues = list()
            self._de_update_queues = list()
            self._close_update_queues = list()
            for i in range(self.distributed_handler.num_memory_hosts):
                with tf.device("/job:memory/task:{}".format(i)):
                    update_q = tf.FIFOQueue(
                        8, [tf.int32, tf.float32],
                        [(self.config["batch_size"], ),
                         (self.config["batch_size"], )],
                        shared_name="updatequeue{}".format(i))
                    self._update_queues.append(update_q)
                    en_q = update_q.enqueue(self.update_phs)
                    self._en_update_queues.append(en_q)
                    de_q = update_q.dequeue()
                    self._de_update_queues.append(de_q)
                    self._close_update_queues.append(
                        update_q.close(cancel_pending_enqueues=True))

    def _setup_learner(self):
        super(ApexAgent, self)._setup_learner()
        if self.config.get("prioritized_replay", False):
            self._learner2mem_q = mp.Queue(8)
            self._stop_learner2mem_indicator = threading.Event()
            self._learner2mem = ProxySender(
                self._learner2mem_q,
                self.executor,
                self._en_update_queues,
                self.update_phs,
                self._stop_learner2mem_indicator,
                choose_buffer_index=True)
            self._learner2mem.start()

    def _setup_communication(self):
        super(ApexAgent, self)._setup_communication()
        if self.config.get("prioritized_replay", False):
            self._learner2mem_q = mp.Queue(8)
            self._stop_learner2mem_indicator = threading.Event()
            self._learner2mem = ProxyReceiver(
                self.executor,
                self._de_update_queues[self.distributed_handler.task_index],
                self._learner2mem_q, self._stop_learner2mem_indicator)
            self._learner2mem.start()

    def _create_buffer(self):
        if self.config.get("prioritized_replay", False):
            buffer = PrioritizedReplayBuffer(
                self.config["buffer_size"],
                self.config["prioritized_replay_alpha"])
        else:
            buffer = ReplayBuffer(self.config["buffer_size"])

        return buffer

    def send_experience(self,
                        obs,
                        actions,
                        rewards,
                        next_obs,
                        dones,
                        weights=None,
                        is_vectorized_env=False,
                        num_env=1):
        """Send collected experience to the memory host(s).
        """
        n_step = self._model_config.get("n_step", 1)
        gamma = self._model_config.get("gamma", 0.99)

        if is_vectorized_env:
            # unstack batch data
            obs_ = np.swapaxes(np.asarray(obs), 0, 1)
            dones_ = np.swapaxes(np.asarray(dones), 0, 1)
            rewards_ = np.swapaxes(np.asarray(rewards), 0, 1)
            actions_ = np.swapaxes(np.asarray(actions), 0, 1)
            next_obs_ = np.swapaxes(np.asarray(next_obs), 0, 1)

            if weights is not None:
                weights_ = np.swapaxes(np.asarray(weights), 0, 1)

            obs_list, actions_list, rewards_list, next_obs_list, dones_list, weights_list = list(), \
                list(), list(), list(), list(), list()
            for i in range(num_env):
                _obs, _actions, _rewards, _next_obs, _dones = n_step_adjustment(
                    obs_[i], actions_[i], rewards_[i], next_obs_[i], dones_[i],
                    gamma, n_step)
                obs_list.append(_obs)
                actions_list.append(_actions)
                rewards_list.append(_rewards)
                next_obs_list.append(_next_obs)
                dones_list.append(_dones)
                if weights is not None:
                    weights_list.append(weights_[i][:len(_obs)])

            obs_stack = np.concatenate(obs_list, axis=0)
            actions_stack = np.concatenate(actions_list, axis=0)
            rewards_stack = np.concatenate(rewards_list, axis=0)
            next_obs_stack = np.concatenate(next_obs_list, axis=0)
            dones_stack = np.concatenate(dones_list, axis=0)

            if weights is not None:
                weights_stack = np.stack(weights_list, axis=0)
            else:
                weights_stack = np.ones(len(rewards_stack), dtype=np.float32)
        else:
            obs_stack, actions_stack, rewards_stack, next_obs_stack, dones_stack = n_step_adjustment(
                obs, actions, rewards, next_obs, dones, gamma, n_step)
            weights_stack = weights or np.ones(
                len(rewards_stack), dtype=np.float32)
            weights_stack = np.asarray(weights_stack[:len(rewards_stack)])
        try:
            self._actor2mem_q.put(
                [
                    arr[:self.config["sample_batch_size"] * num_env]
                    for arr in [
                        obs_stack, actions_stack, rewards_stack,
                        next_obs_stack, dones_stack, weights_stack
                    ]
                ],
                timeout=30)
        except Queue.Full as e:
            logger.warn(
                "actor2mem thread has not sent even one batch for 30 seconds. It is necessary to increase the number of memory hosts."
            )
        finally:
            pass

        self._ready_to_send = False

        # clear the lists
        del obs[:len(obs) - n_step + 1]
        del actions[:len(actions) - n_step + 1]
        del rewards[:len(rewards) - n_step + 1]
        del next_obs[:len(next_obs) - n_step + 1]
        del dones[:len(dones) - n_step + 1]
        if weights is not None:
            del weights[:len(weights) - n_step + 1]

    def communicate(self):
        """Run this method on memory hosts

        Receive transitions from actors and add the data to replay buffers.
        Sample from the replay buffers and send the samples to learners.
        """

        if not self._actor2mem_q.empty():
            samples = self._actor2mem_q.get()
            obs, actions, rewards, next_obs, dones, weights = samples

            self._buffer.add(
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                weights=None)
            self._act_count += np.shape(rewards)[0]
            self._receive_count += np.shape(rewards)[0]
            if int(self._receive_count / 10000) > self._last_receive_record:
                self._last_receive_record = int(self._receive_count / 10000)
                self.executor.run(self._update_num_sampled_timesteps, {})

        if self._act_count >= max(0, self.config["learning_starts"]
                                  ) and not self._mem2learner_q.full():
            if isinstance(self._buffer, PrioritizedReplayBuffer):
                batch_data = self._buffer.sample(
                    self.config["batch_size"],
                    self.config["prioritized_replay_beta"])
                obs, actions, rewards, next_obs, dones, weights, indexes = batch_data["obs"], \
                    batch_data["actions"], batch_data["rewards"], batch_data["next_obs"], batch_data["dones"], \
                    batch_data["weights"], batch_data["indexes"]
                self._mem2learner_q.put(
                    [obs, actions, rewards, next_obs, dones, weights, indexes])
            else:
                batch_data = self._buffer.sample(self.config["batch_size"])
                obs, actions, rewards, next_obs, dones = batch_data["obs"], \
                    batch_data["actions"], batch_data["rewards"], batch_data["next_obs"], batch_data["dones"]
                weights = np.ones_like(rewards)
                self._mem2learner_q.put(
                    [obs, actions, rewards, next_obs, dones, weights])

        if self.config.get("prioritized_replay", False):
            while not self._learner2mem_q.empty():
                data = self._learner2mem_q.get()
                indexes, td_error = data
                new_priorities = (np.abs(td_error) + 1e-6)
                self._buffer.update_priorities(indexes, new_priorities)

    def receive_experience(self):
        """Try to receive collected experience from the memory host(s).
        """
        if self._receive_q.empty():
            return None
        # one queue for one thread and thus there is no racing case.
        # once there exists one batch, the calling of `get` method
        # won't be deadly blocked.
        samples = self._receive_q.get()
        if self.config.get("prioritized_replay", False):
            buffer_id, obs, actions, rewards, next_obs, dones, weights, indexes = samples
            return dict(
                buffer_id=buffer_id,
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                weights=weights,
                indexes=indexes)
        else:
            buffer_id, obs, actions, rewards, next_obs, dones, weights = samples
            return dict(
                buffer_id=buffer_id,
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                weights=np.ones(rewards.shape, dtype=np.float32))

    def learn(self, batch_data):
        """Update upon a batch and send the td_errors to memories if needed

        Returns:
            extra_results (dict): contains the fields computed during an update.
        """
        buffer_id = batch_data.pop("buffer_id", 0)
        extra_results = super(ApexAgent, self).learn(
            batch_data, is_chief=self.distributed_handler.is_chief)
        extra_results["buffer_id"] = buffer_id

        if self.config.get("prioritized_replay",
                           False) and not self._learner2mem_q.full():
            try:
                self._learner2mem_q.put(
                    [
                        int(buffer_id), batch_data["indexes"],
                        extra_results["td_error"]
                    ],
                    timeout=30)
            except Queue.Full as e:
                logger.warn(
                    "learner2mem thread has not sent even one batch for 30 seconds. It is necessary to increase the number of memory hosts."
                )
            finally:
                pass

        return extra_results

    def _init_act_count(self):

        self._act_count = -self._model_config.get("n_step", 1) + 1
