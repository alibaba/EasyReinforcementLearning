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
import os
from datetime import datetime
from tensorflow.core.util.event_pb2 import SessionLog


class UpdateSummarySaverHook(tf.train.SummarySaverHook):
    """A hook to handle TensorFlow summaries

    Run summaries whose values are available at the current `session.run()`.
    Attributes:
        (jones.wz) TO DO: support fine-grained control
    """

    def __init__(self, executor, global_step, job_name, task_index, *args,
                 **kwargs):
        super(UpdateSummarySaverHook, self).__init__(*args, **kwargs)
        self.model = executor
        self._global_step = global_step
        self.job_name = job_name
        self.task_index = task_index

    def begin(self):
        if self.task_index != 0 or self.job_name in ["ps", "memory"]:
            return
        else:
            super(UpdateSummarySaverHook, self).begin()

    def before_run(self, run_context):
        self._request_summary = run_context.original_args[1] is not None and \
            self.model.do_summary and self.task_index == 0 and self.job_name in ["actor", "learner", "worker"] and\
            (self._next_step is None or self._timer.should_trigger_for_step(self._next_step))
        requests = {'global_step': self._global_step}
        if self._request_summary:
            fetches = run_context.original_args[0]
            feed_dict = run_context.original_args[1]
            if isinstance(fetches, list) and isinstance(
                    fetches[-1], dict) and "summary_flag" in fetches[-1]:
                summary_flag = feed_dict[fetches[-1]["summary_flag"]]
                summary_op = self._summary_op.get(summary_flag, None)
                if summary_op is not None:
                    requests['summary'] = [summary_op]
        return tf.train.SessionRunArgs(requests)

    def after_run(self, run_context, run_values):
        if not self._summary_writer:
            return

        stale_global_step = run_values.results["global_step"]
        global_step = stale_global_step + 1
        if self._next_step is None or self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)

        if self._next_step is None:
            self._summary_writer.add_session_log(
                SessionLog(status=SessionLog.START), global_step)

        if "summary" in run_values.results:
            self._timer.update_last_triggered_step(global_step)
            for summary in run_values.results["summary"]:
                self._summary_writer.add_summary(summary, global_step)
            self._summary_writer.flush()

        self._next_step = global_step + 1
