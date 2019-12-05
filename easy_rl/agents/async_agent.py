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

import tensorflow as tf

import sys
import easy_rl.models as models
from easy_rl.agents import ActorLearnerAgent
import numpy as np


class AsyncAgent(ActorLearnerAgent):
    """Actors and learners  exchange data and model parameters in a asynchronous way.

    E.g., ApeX, Impala, A3C (each worker functions as both actor and learner)
    """

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
            is_replica = (self.distributed_handler.job_name == "actor")
            self.model = model_class(
                self.executor.ob_ph_spec,
                self.executor.action_ph_spec,
                model_config=model_config,
                is_replica=is_replica,
                sync_replica_spec={})
            # get the order of elements defined in `learn_feed` which return an OrderedDict
            self._element_names = self.model.learn_feed.keys()

            if self.distributed_handler.job_name == "actor":
                # A exact copy of the global vars used to sync with learner.
                # Thus, there is no need to use gpu, say that get rid of
                # `cuda.host2device`
                with tf.device("/job:actor/task:{}/cpu".format(
                        self.distributed_handler.task_index)):
                    self._behavior_model = model_class(
                        self.executor.ob_ph_spec,
                        self.executor.action_ph_spec,
                        model_config=model_config,
                        is_replica=True,
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
            job_name=self.distributed_handler.job_name,
            task_index=self.distributed_handler.task_index)

        super(AsyncAgent, self)._init()
