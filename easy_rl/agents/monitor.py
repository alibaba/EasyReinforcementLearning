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

import sys, threading, time
import numpy as np
import tensorflow as tf

import logging

logger = logging.getLogger(__name__)


class StopMonitor(threading.Thread):
    def __init__(self, executor, num_sampled_timesteps, in_queue_size,
                 out_queue_size, scheduled_timesteps, scheduled_global_steps,
                 signal):
        threading.Thread.__init__(self)

        self._executor = executor
        self._num_sampled_timesteps = num_sampled_timesteps
        self._global_step = tf.train.get_or_create_global_step()
        self._in_queue_size = in_queue_size
        self._out_queue_size = out_queue_size
        self._scheduled_timesteps = scheduled_timesteps
        self._scheduled_global_steps = scheduled_global_steps
        self._signal = signal

    def run(self):
        while True:
            cur_num_timesteps, cur_global_step, in_q_size, out_q_size = self._executor.run(
                [
                    self._num_sampled_timesteps, self._global_step,
                    self._in_queue_size, self._out_queue_size
                ], {})
            logger.info(
                "cur_num_timesteps:{}, cur_global_step:{}, in_q_size:{}, out_q_size:{}".
                format(cur_num_timesteps, cur_global_step, in_q_size,
                       out_q_size))
            if (cur_num_timesteps >= self._scheduled_timesteps) or (
                    cur_global_step >= self._scheduled_global_steps):
                logger.info(
                    "current_timesteps/max_timesteps:{}/{}, current_global_step/max_global_steps:{}/{} and stop"
                    .format(cur_num_timesteps, self._scheduled_timesteps,
                            cur_global_step, self._scheduled_global_steps))
                self._signal.set()
                break
            time.sleep(1)


class ProxyReceiver(threading.Thread):
    def __init__(self, executor, op, tunnel, signal):
        threading.Thread.__init__(self)

        self._executor = executor
        self._op = op
        self._tunnel = tunnel
        self._signal = signal

    def run(self):
        while not self._signal.is_set():
            if not self._tunnel.full():
                # one queue for one thread, and thus there is no racing case.
                # once there exists one batch, the calling of `get` method
                # won't be deadly blocked.
                try:
                    data = self._executor.run(self._op, {})
                except tf.errors.OutOfRangeError as e:
                    logger.warn(e.message)
                    data = None
                except tf.errors.CancelledError as e:
                    logger.warn(e.message)
                    data = None
                finally:
                    if data:
                        self._tunnel.put(data)


class ProxySender(threading.Thread):
    def __init__(self,
                 tunnel,
                 executor,
                 op,
                 phs,
                 signal,
                 send_buffer_index=None,
                 choose_buffer_index=False):
        threading.Thread.__init__(self)

        self._tunnel = tunnel
        self._executor = executor
        self._op = op
        self._phs = phs
        self._signal = signal
        self._send_buffer_index = send_buffer_index
        self._choose_buffer_index = choose_buffer_index

    def run(self):
        while not self._signal.is_set():
            # one queue for one thread and thus there is no racing cases.
            # once there exists one batch, the calling of `get` method
            # won't be deadly blocked.
            if not self._tunnel.empty():
                data = self._tunnel.get()
                target_buffer_id = None
                if self._send_buffer_index is not None:
                    data.insert(0, self._send_buffer_index)
                elif self._choose_buffer_index:
                    target_buffer_id = data.pop(0)

                try:
                    if isinstance(self._op, list):
                        if target_buffer_id is not None:
                            logger.info(
                                "send to buffer:{}".format(target_buffer_id))
                            fetches = self._op[target_buffer_id]
                        else:
                            fetches = np.random.choice(self._op)
                    else:
                        fetches = self._op

                    feed_dict = dict(
                        (ph, val) for ph, val in zip(self._phs, data))
                    self._executor.run(fetches, feed_dict=feed_dict)
                    logger.info("send data")
                except tf.errors.CancelledError as e:
                    logger.warn(e.message)
                except tf.errors.OutOfRangeError as e:
                    logger.warn(e.message)
                finally:
                    pass
