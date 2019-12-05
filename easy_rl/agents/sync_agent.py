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
import threading, queue
import logging
import numpy as np
import tensorflow as tf

from easy_rl.agents.monitor import ProxyReceiver
from easy_rl.agents import ActorLearnerAgent
from easy_rl.models import models
from easy_rl.utils.buffer import AggregateBuffer

logger = logging.getLogger(__name__)


class SyncAgent(ActorLearnerAgent):
    """Actors and learners  exchange data and model parameters in a synchronous way.

    For on-policy algorithms, e.g., D-PPO and ES.
    """

    def _init(self, model_config, ckpt_dir, custom_model):
        assert self.distributed_handler.num_learner_hosts == 1, "SyncAgent only support one learner currently"
        self.config[
            "batch_size"] = self.distributed_handler.num_actor_hosts * self.config["sample_batch_size"]
        self._setup_sync_barrier()

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
            is_replica = (self.distributed_handler.job_name in ["actor"])
            self.model = model_class(
                self.executor.ob_ph_spec,
                self.executor.action_ph_spec,
                model_config=model_config,
                is_replica=is_replica)
            # get the order of elements defined in `learn_feed` which return an OrderedDict
            self._element_names = self.model.learn_feed.keys()

            if self.distributed_handler.job_name in ["actor"]:
                with tf.device("/job:{}/task:{}/cpu".format(
                        self.distributed_handler.job_name,
                        self.distributed_handler.task_index)):
                    self._behavior_model = model_class(
                        self.executor.ob_ph_spec,
                        self.executor.action_ph_spec,
                        model_config=model_config,
                        is_replica=is_replica,
                        scope='behavior')
            else:
                self._behavior_model = self.model

            self._build_communication(
                job_name=self.distributed_handler.job_name,
                task_index=self.distributed_handler.task_index)

        global_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        if is_replica:
            global_vars = [
                v for v in global_vars
                if v not in self.behavior_model.all_variables
            ]

        self.executor.setup(
            self.distributed_handler.master,
            self.distributed_handler.is_chief,
            self.model.global_step,
            ckpt_dir,
            self.model.summary_ops,
            global_vars=global_vars,
            local_vars=self.behavior_model.all_variables
            if is_replica else None,
            save_var_list=[
                var for var in global_vars if var not in [
                    self._learner_done_flags, self._actor_done_flags,
                    self._ready_to_exit
                ]
            ],
            save_steps=10,
            job_name=self.distributed_handler.job_name,
            task_index=self.distributed_handler.task_index,
            async_mode=False)

        super(SyncAgent, self)._init()

    def _create_buffer(self):
        return AggregateBuffer(self.config.get("buffer_size", 10000))

    def _setup_sync_barrier(self):
        """setup sync barrier for actors, in order to ensure the order of execution
        between actors and learner.
        """
        with self.distributed_handler.get_replica_device():
            self._global_num_sampled_per_iteration = tf.get_variable(
                name="global_num_sampled_per_iteration",
                dtype=tf.int64,
                shape=())

        self._update_global_num_sampled_per_iteration = tf.assign_add(
            self._global_num_sampled_per_iteration,
            np.int64(self.config["sample_batch_size"]),
            use_locking=True)

        self._reset_global_num_sampled_per_iteration = tf.assign(
            self._global_num_sampled_per_iteration,
            np.int64(0),
            use_locking=True)

        self._actor_barrier_q_list = []
        for i in range(self.distributed_handler.num_actor_hosts):
            with tf.device("/job:actor/task:{}".format(i)):
                self._actor_barrier_q_list.append(
                    tf.FIFOQueue(
                        self.distributed_handler.num_actor_hosts,
                        dtypes=[tf.bool],
                        shapes=[()],
                        shared_name="actor_barrier_q{}".format(i)))

        if self.distributed_handler.job_name == "learner":
            self._en_actor_barrier_q_list = [
                e.enqueue(tf.constant(True, dtype=tf.bool))
                for e in self._actor_barrier_q_list
            ]
            self._en_actor_barrier_q_op = tf.group(
                *self._en_actor_barrier_q_list)

        elif self.distributed_handler.job_name == "actor":
            self._de_actor_barrier_q_list = [
                e.dequeue_many(self.distributed_handler.num_learner_hosts)
                for e in self._actor_barrier_q_list
            ]
            self._de_actor_barrier_q_op = self._de_actor_barrier_q_list[
                self.distributed_handler.task_index]
            self._close_actor_barrier_q_list = [
                barrier_q.close(cancel_pending_enqueues=True)
                for barrier_q in self._actor_barrier_q_list
            ]

    def learn(self, batch_data):
        extra_results = super(SyncAgent, self).learn(batch_data)
        self.executor.run([
            self._en_actor_barrier_q_op,
            self._reset_global_num_sampled_per_iteration
        ], {})
        return extra_results

    def _setup_actor(self):
        super(SyncAgent, self)._setup_actor()
        if hasattr(self, "_de_actor_barrier_q_op"):
            self._learner2actor_q = queue.Queue(8)
            self._stop_learner2actor_indicator = threading.Event()
            self._learner2actor = ProxyReceiver(
                self.executor, self._de_actor_barrier_q_op,
                self._learner2actor_q, self._stop_learner2actor_indicator)
            self._learner2actor.start()

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
                time.sleep(5)
                self.executor.session.close()
                logger.info("session closed.")
                return True
            if self.distributed_handler.task_index == 0 and not ready_to_exit:
                # notify actors and learners to exit first
                self.executor.run([self._set_ready_to_exit], {})
        return False

    def _should_learner_stop(self):
        ready_to_exit = self.executor.run(self._ready_to_exit, {})
        if ready_to_exit:
            if not self._stop_receiver_indicator.is_set():
                # Learner was notified at the first time.
                # Notify the receiver thread to stop but the main thread
                # continues
                self._stop_receiver_indicator.set()
            else:
                if not self._receiver.is_alive():
                    # The receiver thead has left `run()` method.
                    # Threads are allowed to be joined for more than
                    # once.
                    self._receiver.join()
                    logger.info("threads joined.")
                    # See if there still data need to be consumed
                    # As the thread (i.e., the producer) has done and
                    # consumer is this main thread itself, there is no
                    # inconsistent issue here.
                    if self._receive_q.empty():
                        if self.distributed_handler.task_index == 0:
                            # chief worker (i.e., learner_0) is responsible for exporting
                            # saved_model.
                            self.export_saved_model()
                        self.executor.run(self._set_stop_flag, {})
                        should_stop = False
                        while not should_stop:
                            should_stop = self.executor.run(
                                self._should_stop, {})
                        logger.info("all actors and learners have done.")
                        self.executor.session.close()
                        logger.info("session closed.")
                        return should_stop
        return False

    def _should_actor_stop(self):
        ready_to_exit = self.executor.run(self._ready_to_exit, {})
        if ready_to_exit and self._stop_sender_indicator.is_set():
            self._actor2mem.join()
            logger.info("actor2mem thread joined.")
            if not self._learner2actor.is_alive():
                # Menas that we have executed the following code snippet
                return True
            # close the sync barrier queue.
            self.executor.run(
                self._close_actor_barrier_q_list[
                    self.distributed_handler.task_index], {})
            self._stop_learner2actor_indicator.set()
            self._learner2actor.join()
            logger.info("learner2actor thread joined.")
            # notify the memory
            self.executor.run(self._set_stop_flag, {})
            should_stop = False
            while not should_stop:
                should_stop = self.executor.run(self._should_stop, {})
            logger.info("all actors and learners have done.")
            time.sleep(5)
            self.executor.session.close()
            logger.info("session closed.")
            return should_stop
        return False
