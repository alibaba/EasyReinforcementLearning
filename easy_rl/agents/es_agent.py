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

import time
import six.moves.queue as Queue

import datetime
import numpy as np
from easy_rl.models import EvolutionStrategy
from easy_rl.agents import SyncAgent

import logging

logger = logging.getLogger(__name__)


class ESAgent(SyncAgent):
    """ES is an on-policy stochastic optimization method

    Actors are tasked to generate perturbations and evaluate the performances of perturbed model parameters.
    The learner aggregates the perturbations into one update direction based on their performances.
    """

    _agent_name = "ES"
    _default_model_class = "ES"
    _valid_model_classes = [EvolutionStrategy]
    _valid_model_class_names = ["ES"]

    def send_experience(self, **kwargs):
        try:
            self._actor2mem_q.put([
                arr[:self.config["sample_batch_size"]] for arr in (
                    [np.asarray(kwargs[name]) for name in self._element_names])
            ])
        except Queue.Full as e:
            logger.warning("{}".format(e))
        finally:
            pass
        current_sampled = self.executor.run(
            self._update_global_num_sampled_per_iteration, {})

        # waiting for learner optimization then start the next iteration
        logger.info("curent_sampled:{}".format(current_sampled))
        logger.info("sync at {}".format(datetime.datetime.now()))

        got_signal = False
        while not got_signal:
            # See if the learner has prepared for leaving.
            learner_done_flags = self.executor.run(self._learner_done_flags,
                                                   {})
            # Assume there is only one learner which is the case of DPPO.
            if learner_done_flags[0]:
                # Notify the `should_stop()` to stop.
                self._stop_sender_indicator.set()
                break

            # Try to receive sync barrier signal as usual.
            try:
                sync_barrier_signal = self._learner2actor_q.get(timeout=10)
                got_signal = True
            except Queue.Empty as e:
                logger.debug("wait for sync barrier signal for 10 seconds.")
                got_signal = False
            finally:
                pass

        logger.info("sync end at {}".format(datetime.datetime.now()))
        self._ready_to_send = False

        # clear the lists
        for k in kwargs.keys():
            del kwargs[k][:len(kwargs[k])]

    def communicate(self):
        """Run this method on memory hosts

        Receive transitions from actors and add the data to replay buffers.
        Sample from the replay buffers and send the samples to learners.
        """
        if not self._actor2mem_q.empty():
            samples = self._actor2mem_q.get()

            self._buffer.add(**{
                name: value
                for name, value in zip(self._element_names, samples)
            })
            self._receive_count += np.shape(samples[0])[0]
            if int(self._receive_count / 10000) > self._last_receive_record:
                self._last_receive_record = int(self._receive_count / 10000)
                self.executor.run(self._update_num_sampled_timesteps, {})
            logger.info("memory add:{}".format(self._receive_count))

            if self._receive_count % (
                    self.config["sample_batch_size"] *
                    self.distributed_handler.num_actor_hosts) == 0:
                batch_data = self._buffer.sample(
                    self.config["sample_batch_size"] *
                    self.distributed_handler.num_actor_hosts)
                self._mem2learner_q.put(
                    [batch_data[name] for name in self._element_names])

    def receive_experience(self):
        if self._receive_q.empty():
            return None
        samples = self._receive_q.get()
        # drop the buffer_id
        samples.pop(0)
        logger.info("learner receive:{}".format(len(samples[-1])))
        return {
            name: value
            for name, value in zip(self._element_names, samples)
        }
