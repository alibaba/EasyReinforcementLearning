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


class DistributedHandler(object):
    """Class for handling distributed Tensorflow

    Provide device context for building graph and arguments for TensorFlow sessions.
    Attributes:
        master (obj): used to specify the `master` argument of `MonitoredTrainingSession`.
        server (obj): an optional attribute which is needed under distributed setting.
        job_name (str): "ps", "memory", "actor", "learner", or "" which indicates the single-machine setting.
        task_index (int): task index.
        is_chief (bool): used to specify the `is_chief` argument of `MonitoredTrainingSession`.
        device (obj): the proper device context for `Model` object to place ops.
        num_memory_hosts (int): the number of hosts play a memory role.
        num_actor_hosts (int): the number of hosts play an actor role.
        num_learner_hosts (int): the number of hosts play a learner role.
        cluster (obj): optional. provided under a distributed setting.
    """

    def __init__(self, config):
        """Construct an DistributedHandler object.

        Arguments:
            config (dict): specifying distributed computation.

            e.g., dict(
                job_name="actor", # (taking a value from {"actor", "learner", "memory", "ps"})
                task_id=0,
                ps_hosts: "ip,ip"
                learner_hosts: "ip,ip,ip"
                memory_hosts: "ip,ip"
                actor_hosts: "ip,ip,ip,ip"
            )
        """

        config = config or {}

        for k in config.keys():
            if k.endswith("_hosts"):
                config[k] = config[k].split(',')

        # Situations where the server has been created and the device
        # context has been entered, e.g., Alibaba Porsche TensorFlowOnFlink
        if "target" in config:
            # (jones.wz) TO DO: the scripts are running on workers which makes "memory" role unavailable.
            self.master = config["target"]
            self.job_name = "worker"
            self.task_index = config["task_index"]
            self.is_chief = (config["task_index"] == 0)
            self.device = tf.device("/cpu")
        else:
            if config.get("ps_hosts", ''):
                # In a distributed setting
                self.cluster = tf.train.ClusterSpec(
                    dict((k[:-len("_hosts")], v) for k, v in config.items()
                         if k.endswith("_hosts")))
                self.server = tf.train.Server(
                    server_or_cluster_def=self.cluster,
                    job_name=config["job_name"],
                    task_index=config["task_index"],
                    protocol=None,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)),
                    start=True)
                self.master = self.server.target
                self.job_name = config["job_name"]
                self.task_index = config["task_index"]
                self.is_chief = (config["task_index"] == 0 and
                                 config["job_name"] in ["learner", "worker"])
                self.device = tf.device(
                    tf.train.replica_device_setter(
                        worker_device="/job:" + config["job_name"] +
                        ("/task:%d" % config["task_index"]),
                        cluster=self.cluster))
                self.num_memory_hosts = len(config.get("memory_hosts", []))
                self.num_actor_hosts = len(config.get("actor_hosts", []))
                self.num_learner_hosts = len(config.get("learner_hosts", []))
            else:
                # Single-machine
                self.master = ''
                self.job_name = ''
                self.task_index = 0
                self.is_chief = True
                self.device = tf.device("/cpu")

    def get_replica_device(self):
        return tf.device(
            tf.train.replica_device_setter(
                worker_device="/job:" + self.job_name +
                ("/task:%d" % self.task_index),
                cluster=self.cluster))
