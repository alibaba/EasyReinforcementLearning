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
import logging
import threading
import queue, time

import numpy as np
import tensorflow as tf

from easy_rl.agents.monitor import StopMonitor, ProxyReceiver, ProxySender
from easy_rl.agents import AgentBase

logger = logging.getLogger(__name__)


class ActorLearnerAgent(AgentBase):
    """Actor-learner architecture.

    We regard some workers as learners and some as actors.
    Meanwhile, some ps do the job of parameter servers and some ps work for caching data.
    We declare model parameters on both the local host and the parameter servers.
    Learner push changes of model parameters to the servers.
    Actors pull the latest model parameters from the servers.
    """

    def _init(self):
        if self.distributed_handler.job_name == "memory":
            self._setup_communication()
        elif self.distributed_handler.job_name == "learner":
            self._setup_learner()
        elif self.distributed_handler.job_name == "actor":
            self._setup_actor()
        # As a distributed computing paradigm, we need complicated mechanism
        # (encapsulated in `should_stop()` method) for making decisions on
        # when to stop. Like a context object, once stopped, the instance
        # would not work normally anymore.
        self._stopped = False

    def _setup_actor(self):
        self._send_cost = self._put_cost = 0
        self._actor_mem_cost = [0, 0]
        self._actor2mem_q = queue.Queue(8)
        self._stop_sender_indicator = threading.Event()
        self._actor2mem = ProxySender(self._actor2mem_q, self.executor,
                                      self._en_in_queues, self.in_queue_phs,
                                      self._stop_sender_indicator)
        self._actor2mem.start()

    def _setup_learner(self):
        # create threads for non-blocking communication
        self._receive_q = queue.Queue(8)
        self._stop_receiver_indicator = threading.Event()
        self._receiver = ProxyReceiver(
            self.executor,
            self._de_out_queues[self.distributed_handler.task_index % len(
                self._out_queues)], self._receive_q,
            self._stop_receiver_indicator)
        self._receiver.start()

    def _create_buffer(self):
        """create buffer according to the specific model"""
        raise NotImplementedError

    def _setup_communication(self):
        # Under single machine setting, we create buffer object as the class attribute
        # The type of buffer should be determined by the model type
        self._buffer = self._create_buffer()

        self._stop_indicator = threading.Event()
        # create a thread for monitoring the training courses
        self._monitor = StopMonitor(
            self.executor, self._num_sampled_timesteps, self._in_queue_size,
            self._out_queue_size,
            self.config.get("scheduled_timesteps", 1000000),
            self.config.get("scheduled_global_steps",
                            1000), self._stop_indicator)
        self._monitor.start()

        # create threads for non-blocking communication
        self._actor2mem_q = queue.Queue(8)
        self._stop_actor2mem_indicator = threading.Event()
        self._actor2mem = ProxyReceiver(
            self.executor,
            self._de_in_queues[self.distributed_handler.task_index],
            self._actor2mem_q, self._stop_actor2mem_indicator)
        self._actor2mem.start()

        self._mem2learner_q = queue.Queue(8)
        self._stop_mem2learner_indicator = threading.Event()
        self._mem2learner = ProxySender(
            self._mem2learner_q,
            self.executor,
            self._en_out_queues[self.distributed_handler.task_index % len(
                self._out_queues)],
            self.out_queue_phs,
            self._stop_mem2learner_indicator,
            send_buffer_index=self.distributed_handler.task_index)
        self._mem2learner.start()

    def _get_in_queue_meta(self):
        """Determine the type, shape, and input placeholders for in_queue (actor --> memory)
        """
        phs = list()
        dtypes = list()
        shapes = list()
        num_env = self.config.get("num_env", 1)
        for name in self._element_names:
            v = self.model.learn_feed[name]
            if name in ["obs", "next_obs"]:
                ph = tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.config["sample_batch_size"] * num_env, ) +
                    self.executor.flattened_ob_shape)
            else:
                ph = tf.placeholder(
                    dtype=v.dtype,
                    shape=[self.config["sample_batch_size"] * num_env] +
                    v.shape.as_list()[1:])
            phs.append(ph)
            dtypes.append(ph.dtype)
            shapes.append(ph.shape)
        return dtypes, shapes, phs

    def _get_out_queue_meta(self):
        """Determine the type, shape, and input placeholders for out_queue (memory --> learner)
        """
        phs = list()
        dtypes = list()
        shapes = list()

        # add index of memory
        mem_index_ph = tf.placeholder(dtype=tf.int32, shape=())
        phs.append(mem_index_ph)
        dtypes.append(mem_index_ph.dtype)
        shapes.append(mem_index_ph.shape)

        for name in self._element_names:
            v = self.model.learn_feed[name]
            if name in ["obs", "next_obs"]:
                ph = tf.placeholder(
                    dtype=tf.float32,
                    shape=(self.config["batch_size"], ) +
                    self.executor.flattened_ob_shape)
            else:
                ph = tf.placeholder(
                    dtype=v.dtype,
                    shape=[self.config["batch_size"]] + v.shape.as_list()[1:])
            phs.append(ph)
            dtypes.append(ph.dtype)
            shapes.append(ph.shape)
        return dtypes, shapes, phs

    def _build_communication(self, job_name, task_index):
        """Build the subgraph for communication between actors, memories, and learners
        """
        if job_name in ["actor", "memory"]:
            # data flow: actor --> memory
            dtypes, shapes, self.in_queue_phs = self._get_in_queue_meta()
            self._in_queues = list()
            self._en_in_queues = list()
            self._de_in_queues = list()
            self._close_in_queues = list()
            for i in range(self.distributed_handler.num_memory_hosts):
                with tf.device("/job:memory/task:{}".format(i)):
                    in_q = tf.FIFOQueue(
                        8, dtypes, shapes, shared_name="inqueue{}".format(i))
                    self._in_queues.append(in_q)
                    en_q = in_q.enqueue(self.in_queue_phs)
                    self._en_in_queues.append(en_q)
                    de_q = in_q.dequeue()
                    self._de_in_queues.append(de_q)
                    self._close_in_queues.append(
                        in_q.close(cancel_pending_enqueues=True))
            self._in_queue_size = self._in_queues[
                self.distributed_handler.task_index % len(
                    self._in_queues)].size()

        # data flow: memory --> learner
        dtypes, shapes, self.out_queue_phs = self._get_out_queue_meta()
        self._out_queues = list()
        self._en_out_queues = list()
        self._de_out_queues = list()
        self._close_out_queues = list()
        if job_name == "memory":
            for i in range(self.distributed_handler.num_learner_hosts):
                with tf.device("/job:learner/task:{}".format(i)):
                    out_q = tf.FIFOQueue(
                        8, dtypes, shapes, shared_name="outqueue{}".format(i))
                    self._out_queues.append(out_q)
                    en_q = out_q.enqueue(self.out_queue_phs)
                    self._en_out_queues.append(en_q)
                    de_q = out_q.dequeue()
                    self._de_out_queues.append(de_q)
                    self._close_out_queues.append(
                        out_q.close(cancel_pending_enqueues=True))
            self._out_queue_size = self._out_queues[
                self.distributed_handler.task_index % len(
                    self._out_queues)].size()

        if job_name == "learner":
            with tf.device("/job:learner/task:{}".format(
                    self.distributed_handler.task_index)):
                out_q = tf.FIFOQueue(
                    8,
                    dtypes,
                    shapes,
                    shared_name="outqueue{}".format(
                        self.distributed_handler.task_index))
                self._out_queues.append(out_q)
                en_q = out_q.enqueue(self.out_queue_phs)
                self._en_out_queues.append(en_q)
                de_q = out_q.dequeue()
                self._de_out_queues.append(de_q)
                self._close_out_queues.append(
                    out_q.close(cancel_pending_enqueues=True))

        # create an op for actors to obtain the latest vars
        sync_var_ops = list()
        for des, src in zip(self.behavior_model.actor_sync_variables,
                            self.model.actor_sync_variables):
            sync_var_ops.append(tf.assign(des, src))
        self._sync_var_op = tf.group(*sync_var_ops)

        # create some vars and queues for monitoring the training courses
        self._num_sampled_timesteps = tf.get_variable(
            "num_sampled_timesteps", dtype=tf.int64, initializer=np.int64(0))

        self._learner_done_flags = tf.get_variable(
            "learner_done_flags",
            dtype=tf.bool,
            initializer=np.asarray(
                self.distributed_handler.num_learner_hosts * [False],
                dtype=np.bool))
        self._actor_done_flags = tf.get_variable(
            "actor_done_flags",
            dtype=tf.bool,
            initializer=np.asarray(
                self.distributed_handler.num_actor_hosts * [False],
                dtype=np.bool))
        self._should_stop = tf.logical_and(
            tf.reduce_all(self._learner_done_flags),
            tf.reduce_all(self._actor_done_flags))
        if self.distributed_handler.job_name == "learner":
            self._set_stop_flag = tf.assign(
                self._learner_done_flags[self.distributed_handler.task_index],
                np.bool(1),
                use_locking=True)
        if self.distributed_handler.job_name == "actor":
            self._set_stop_flag = tf.assign(
                self._actor_done_flags[self.distributed_handler.task_index],
                np.bool(1),
                use_locking=True)

        self._ready_to_exit = tf.get_variable(
            "global_ready_to_exit", dtype=tf.bool, initializer=np.bool(0))
        self._set_ready_to_exit = tf.assign(
            self._ready_to_exit, np.bool(1), use_locking=True)

        self._update_num_sampled_timesteps = tf.assign_add(
            self._num_sampled_timesteps, np.int64(10000), use_locking=True)

    def sync_vars(self):
        """Sync with the latest vars
        """
        self.executor.run(self._sync_var_op, {})

    def join(self):
        """Call `server.join()` if the agent object serves as a parameter server.
        """

        self.distributed_handler.server.join()

    def communicate(self):
        raise NotImplementedError

    def should_stop(self):
        """Judge whether the agent should stop.
        """
        if not self._stopped:
            if self.distributed_handler.job_name == "memory":
                self._stopped = self._should_memory_stop()
            elif self.distributed_handler.job_name == "learner":
                self._stopped = self._should_learner_stop()
            elif self.distributed_handler.job_name == "actor":
                self._stopped = self._should_actor_stop()
            return self._stopped
            # ps host won't exit
        else:
            return True

    def _should_memory_stop(self):
        if self._stop_indicator.is_set():
            # as the monitor thread has set the event
            self._monitor.join()
            should_stop, ready_to_exit = self.executor.run(
                [self._should_stop, self._ready_to_exit], {})
            if should_stop:
                # need to close the queues so that the threads that
                # execute `session.run()` would not be deadly blocked
                fetches = [self._close_in_queues, self._close_out_queues]
                if hasattr(self, "_close_update_queues"):
                    fetches.append(self._close_update_queues)
                if hasattr(self, "_close_actor_barrier_q_op"):
                    fetches.append(self._close_actor_barrier_q_op)
                self.executor.run(fetches, {})
                # Even though theses threads are running `enqueue` or
                # `dequeue` op, they won't be deadly blocked. Instead,
                # as we have closed the TF FIFOQueues, these threads
                # will throw corresponding exceptions as we expected.
                self._stop_mem2learner_indicator.set()
                self._mem2learner.join()
                self._stop_actor2mem_indicator.set()
                self._actor2mem.join()
                if hasattr(self, "_learner2mem"):
                    self._stop_learner2mem_indicator.set()
                    self._learner2mem.join()
                self.executor.session.close()
                return True
            if self.distributed_handler.task_index == 0 and not ready_to_exit:
                # notify actors and learners to exit first
                self.executor.run([self._set_ready_to_exit], {})
        return False

    def _should_learner_stop(self):
        ready_to_exit = self.executor.run(self._ready_to_exit, {})
        if ready_to_exit:
            self._stop_receiver_indicator.set()
            self._receiver.join()
            if hasattr(self, "_learner2mem"):
                self._stop_learner2mem_indicator.set()
                self._learner2mem.join()
            logger.info("threads joined.")
            if self.distributed_handler.task_index == 0:
                # chief worker (i.e., learner_0) is responsible for exporting
                # saved_model.
                self.export_saved_model()
            self.executor.run(self._set_stop_flag, {})
            should_stop = False
            while not should_stop:
                should_stop = self.executor.run(self._should_stop, {})
            logger.info("all actors and learners have done.")
            if self.distributed_handler.is_chief:
                time.sleep(30)
            self.executor.session.close()
            logger.info("session closed.")
            return should_stop
        return False

    def _should_actor_stop(self):
        ready_to_exit = self.executor.run(self._ready_to_exit, {})
        if ready_to_exit:
            self._stop_sender_indicator.set()
            # If the Queue is full, the thread must not enter its `if`
            # branch and thus will exit `run()` immediately. Otherwise,
            # as memory hosts have not stopped, the enqueue op would not
            # be blocked.
            self._actor2mem.join()
            logger.info("thread joined.")
            # notify the memory
            self.executor.run(self._set_stop_flag, {})
            should_stop = False
            while not should_stop:
                should_stop = self.executor.run(self._should_stop, {})
            logger.info("all actors and learners have done.")
            self.executor.session.close()
            logger.info("session closed.")
            return should_stop
        return False
