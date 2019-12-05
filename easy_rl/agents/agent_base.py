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

import os
import logging
from datetime import datetime
import tensorflow as tf
from easy_rl.agents.executor import Executor
from easy_rl.agents.distributed_handler import DistributedHandler
from easy_rl.utils.schedules import LinearSchedule, ConstantSchedule

logger = logging.getLogger(__name__)


class AgentBase(object):
    """All EasyRL agent classes extend this base class

    Agent objects expose `act()` method to interact with Environment objects
    and `learn()` method to update the underlying `Model` object(s).

    Attributes:
        executor (obj): Handle the runtime.
        model (obj): Provide the computation graph.
        distributed_handler (obj): Provide the context for distributed computing.
        ready_to_send (bool): Whether the agent is ready to send collected experience.
    """

    def __init__(self,
                 observation_space,
                 action_space,
                 agent_config,
                 model_config,
                 distributed_spec,
                 custom_model=None,
                 checkpoint_dir="",
                 export_dir=""):
        """Construct an EasyRL AgentBase object

        Specification of subclasses can be expressed in `_init()` method.

        Arguments:
            observation_space (gym.spaces.Space): specify the shapes and types of observed states.
            action_space (gym.spaces.Space): specify the shapes and types of actions.
            agent_config (dict): specify the agent.
            model_config (dict): specify the underlying model object(s).
            distributed_spec (dict): specify the distributed computation.
            custom_model (cls): user defined model(should be inherited from default model).
            checkpoint_dir (str): specify the checkpoint directory of TF session.
            export_dir (str): specify the default path to export SavedModel.
        """
        self.executor = Executor(observation_space, action_space)
        self.config = agent_config
        self._model_config = model_config
        self.distributed_handler = DistributedHandler(distributed_spec)

        self._export_dir = export_dir

        self._init(model_config, checkpoint_dir, custom_model)

        # control the frequency of reset perturbation
        self._episode_count = 0

        # control the frequency of sending transitions to the buffer
        self._init_act_count()

        # record the number of samples received by memory
        self._receive_count = 0
        self._last_receive_record = 0
        self._sample_batch_size = self.config.get("sample_batch_size", 1)
        self._ready_to_send = False

        # control the frequency of receiving transitions from the buffer
        self._ready_to_receive = False
        # sub batch_size for training
        self._sub_train_batch = self.config.get("sub_train_batch", None)
        # number of epoch to use the given batch_data.
        self._train_epochs = self.config.get("train_epochs", 1)
        self._learn_count = 0
        self._sync_target_count = 0

        # \epsilon value which decays along a given interval.
        # deterministic policy uses \epsilon for \epsilon-greedy exploration
        # (jones.wz) TO DO: handle the cases of stochastic policy
        constant_eps = agent_config.get("constant_eps", None)
        if constant_eps is not None:
            self._exploration = ConstantSchedule(value=constant_eps)
        else:
            self._exploration = LinearSchedule(
                schedule_timesteps=int(
                    agent_config.get("exploration_timesteps", 4e5)),
                initial_val=1.0,
                final_val=agent_config.get("final_eps", .02))

        # kl-threshold used in parameters space noise which decays along a given interval.
        # (huxu.hx) TO DO: need to create a general strategy handle eps and kl?
        self._noise_kl_threshold = LinearSchedule(
            schedule_timesteps=int(agent_config.get("noise_kl_episodes", 4e2)),
            initial_val=0.1,
            final_val=agent_config.get("final_eps", .0001))

    def _init(self, model_config, ckpt_dir, custom_model):
        """Build the computation graph and setup the session according to the specific sub-class

        Instantiate a specific model class to build the corresponding computation graph.
        e.g.,:
            def _init(self, model_config, ckpt_dir):
                with self.distributed_handler.device:
                    self.model = _default_model_class(self.executor.ob_plh_spec,
                                                      self.executor.action_plh_spec,
                                                      model_config)
                    # build queues and related ops for communication
                self.executor.setup()

        Arguments:
            model_config (dict): specify the model to be instantiated.
            ckpt_dir (str): specify the checkpoint directory of TF session.
            custom_model: (cls): user defined model (should be inherited from default model).
        """
        raise NotImplementedError

    @property
    def behavior_model(self):
        return self._behavior_model

    def act(self,
            obs,
            deterministic,
            use_perturbed_action=False,
            eval=False,
            **kwargs):
        """Predict actions given input observations.

        No need for subclasses to override this method.
        Specification can be achieved by specifying different model.

        Arguments:
            obs (np.ndarray): the input observations.
            deterministic (bool): if false, exploration/sampling are conducted.
            use_perturbed_action (bool): set `true` to use perturbed action.
            eval (bool): act in evaluation, act_count increase will be ignored.
            kwargs (dict): extra model-specific data needed for the act operation.

        Returns:
            actions (np.ndarray): predicted actions as a numpy array.
            extra_results (dict): extra outputs.
        """
        if use_perturbed_action:
            assert hasattr(
                self._behavior_model, "perturbed_actions"
            ), "{} model has no atrribute for perturbed_actions".format(
                type(self._behavior_model))
            fetches = [
                self._behavior_model.perturbed_actions,
                self._behavior_model.extra_act_fetches
            ]
        else:
            fetches = [
                self._behavior_model.output_actions,
                self._behavior_model.extra_act_fetches
            ]

        fetches.append({"summary_flag": self.model._set_summary_flag})

        feed_dict = self.executor.feed_observations(
            self._behavior_model.obs_ph, obs)
        feed_dict[self._behavior_model.deterministic_ph] = deterministic
        kwargs["eps"] = self._exploration.value(self._act_count)

        if self._act_count % 10000 == 0:
            logger.info("act_count: {}, eps: {}".format(
                self._act_count, kwargs["eps"]))

        kwargs["noise_kl_threshold"] = self._noise_kl_threshold.value(
            self._episode_count)

        if self._episode_count % self.config.get("perturbation_frequency",
                                                 50) == 0:
            kwargs["reset"] = True

        extra_feed_dcit = {
            v: kwargs[k]
            for k, v in self.behavior_model.extra_act_feed.items()
            if k in kwargs
        }
        feed_dict.update(extra_feed_dcit)
        feed_dict.update({self.model._set_summary_flag: "act"})

        self.executor.do_summary = True
        actions, extra_results, _ = self.executor.run(fetches, feed_dict)
        self.executor.do_summary = False
        # restore the action shape for Box type action space, as they are flattened
        actions = self.executor.restore_action_shape_if_needed(actions)

        # count for the times of taking an action, so that we know when to send collected experience
        if not eval:
            self._act_count += 1
            if self._act_count > 0 and self._act_count % self._sample_batch_size == 0:
                self._ready_to_send = True
        return actions, extra_results

    @property
    def ready_to_send(self):
        return self._ready_to_send

    def send_experience(self, obs, actions, rewards, new_obs, done_masks,
                        **kwargs):
        """Send the experience to the buffer

        Either add the experience to the (replay) beffer object under the single machine setting,
        or send the exeprience to the memory hosts under the distributed setting.

        Arguments:
            obs (obj): the input overvations.
            actions (obj): the actions behaved in the interactions.
            kwargs (dict): extra model-specific data needed for the update operation.
        """
        raise NotImplementedError

    @property
    def ready_to_receive(self):
        return self._ready_to_receive

    def receive_experience(self):
        """Recieve the experience from the buffer
        """
        raise NotImplementedError

    def learn(self, batch_data, is_chief=True, need_recovery=False):
        """Update the model(s) with respect to the input data

        Arguments:
            batch_data (dict): contains the fields for updating models.
            need_recovery (bool): whether to recovery multi-channel obs from flattened array.

        Returns:
            extra_results (dict): extra outputs.
        """

        fetches = [
            self.model.update_op, self.model.extra_learn_fetches, {
                "summary_flag": self.model._set_summary_flag
            }
        ]

        for key in batch_data.keys():
            if key not in ["obs", "next_obs"]:
                batch_data_length = len(batch_data[key])
                break
        train_size = min(self._sub_train_batch or batch_data_length,
                         batch_data_length)
        extra_results = []
        for e in range(self._train_epochs):
            offset = 0

            # number of last samples with less than train_size will be discarded
            while offset <= (batch_data_length - train_size):
                feed_dict = {
                    v: batch_data[k][offset:offset + train_size]
                    for k, v in self.model.learn_feed.items()
                    if k not in ["obs", "next_obs"]
                }
                for field in ["obs", "next_obs"]:
                    if field in self.model.learn_feed.keys():
                        feed_dict.update(
                            self.executor.feed_observations(
                                self.model.learn_feed[field],
                                batch_data[field],
                                train_size=train_size,
                                offset=offset))
                feed_dict.update({self.model._set_summary_flag: "train"})
                self.executor.do_summary = True
                _, extra_result, _ = self.executor.run(fetches, feed_dict)
                self.executor.do_summary = False
                extra_results.append(extra_result)

                offset += train_size
        if len(extra_results) == 1:
            extra_results = extra_results[0]

        self._learn_count += 1
        if hasattr(
                self.model, "sync_target_op"
        ) and self._learn_count % self.config["sync_target_frequency"] == 0 and is_chief:
            self._sync_target_count += 1
            if self._sync_target_count % self.config.get(
                    "sync_logging_freq", 128) == 0:
                logger.info("learn_count: {}, sync_target_count: {}".format(
                    self._learn_count, self._sync_target_count))
            self.executor.run(self.model.sync_target_op, {})

        return extra_results

    def add_episode(self, num_episode):

        self._episode_count += num_episode

    def _init_act_count(self):

        self._act_count = 0

    def add_extra_summary(self, feed_dict):
        """add any data which is out of computation graph into summary events.
        the associated summary op needs to be defined in `model.add_extra_summary_op` in advance.

        Arguments:
            feed_dict (dict): pair of summary_op and data.
        """
        fetches = [{"summary_flag": self.model._set_summary_flag}]
        feed_dict.update({self.model._set_summary_flag: "extra"})

        self.executor.do_summary = True
        self.executor.run(fetches=fetches, feed_dict=feed_dict)
        self.executor.do_summary = False

    def export_saved_model(self, export_dir=""):

        export_dir = export_dir or self._export_dir

        if export_dir:
            self.executor.unsafe_unfinalize()

            if isinstance(self.behavior_model.obs_ph, dict):
                inputs = self.behavior_model.obs_ph
            elif isinstance(self.behavior_model.obs_ph, list):
                inputs = {
                    "obs_ph{}".format(i): ts
                    for i, ts in enumerate(self.behavior_model.obs_ph)
                }
            else:
                inputs = {"obs_ph": self.behavior_model.obs_ph}
            inputs.update({
                "deterministic_ph": self.behavior_model.deterministic_ph
            })
            inputs.update(self.behavior_model.extra_act_feed)

            outputs = dict(output_actions=self.behavior_model.output_actions)
            if hasattr(self.behavior_model, "perturbed_actions"):
                outputs.update({
                    "perturbed_actions": self.behavior_model.perturbed_actions
                })
            outputs.update({
                name: ts
                for name, ts in self.behavior_model.extra_act_fetches.items()
                if not isinstance(ts, type(tf.no_op()))
            })

            signature_def = tf.saved_model.signature_def_utils.predict_signature_def(
                inputs=inputs, outputs=outputs)

            builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

            def get_session(sess):
                session = sess
                while type(session).__name__ != 'Session':
                    session = session._sess
                return session

            builder.add_meta_graph_and_variables(
                get_session(self.executor.session),
                [tf.saved_model.tag_constants.SERVING],
                signature_def_map={"predict_results": signature_def},
                clear_devices=True)

            builder.save()
            self.executor.finalize()
            with tf.gfile.GFile(os.path.join(export_dir, "dump_finish"),
                                "w") as fw:
                fw.write("dump_finish @{}".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
