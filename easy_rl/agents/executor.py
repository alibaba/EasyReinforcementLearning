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
from collections import OrderedDict
import numpy as np
import re, sys, os
import gym.spaces
import tensorflow as tf

from easy_rl.utils.utils import prod
import easy_rl.utils.hooks

logger = logging.getLogger(__name__)


class Executor(object):
    """class that handles the runtime for Agent classes

    Maintain the session obj and its related issues
    Attributes:
        observation_space (obj): gym.spaces.Space object for specifying the shapes and types of input observation(s).
        action_space (obj): gym.spaces.Space object for specifying the shapes and types of the action space.
        _is_single_channel (bool): indicates whether the observation is single-channel.
        ob_ph_spec (obj): parse observation_space to guide `Model` object for building placeholders.
        flattened_ob_shape (int): the shape of each (flattened) observation.
        action_ph_spec (tuple): the type (refers to the action distribution) and dims of action space.
        do_summary (bool): control whether to run summary ops at each `session.run()`.
        session (obj): a TensorFlow MonitoredTrainingSession object.
    """

    def __init__(self, observation_space, action_space):
        """Construct an Executor object

        Create TensorFlow placeholders as the input nodes of Model classes first
        and set up the session later.
        Arguments:
            observation_space (gym.spaces.space obj): specify the shapes and types of observed states.
            action_space (gym.spaces.space obj): specify the shapes and types of actions.
        """

        self.observation_space = observation_space
        self.action_space = action_space
        self._prepare_ph_spec()

    def _prepare_ph_spec(self):
        """Build the TensorFlow placeholders according to the `observation_space`.

        Forbid multi-channel observations where any individual channel is recursively defined as a multi-channel observation.
        `_prepare_ph_spec()` can easily handle recursively defined observations, but they introduce unnecessary complexity to the TensorFlow FIFOqueues used for data exchanging in distributed setting.
        """

        if isinstance(self.observation_space, gym.spaces.Tuple):
            self._is_single_channel = False
            self.ob_ph_spec = list()
            for sp in self.observation_space.spaces:
                assert type(sp) not in [
                    gym.spaces.Tuple, gym.spaces.Dict
                ], "forbidden type {}".format(self.observation_space)
                self.ob_ph_spec.append(self._basic_space_to_ph_spec(sp))
            self.flattened_ob_shape = (np.sum(
                [s[1][1] for s in self.ob_ph_spec]), )
        elif isinstance(self.observation_space, gym.spaces.Dict):
            self._is_single_channel = False
            self.ob_ph_spec = OrderedDict()
            for sp_name, sp in self.observation_space.spaces.items():
                assert type(sp) not in [
                    gym.spaces.Tuple, gym.spaces.Dict
                ], "forbidden type {}".format(self.observation_space)
                self.ob_ph_spec[sp_name] = self._basic_space_to_ph_spec(sp)
            self.flattened_ob_shape = (np.sum(
                [s[1][1] for s in self.ob_ph_spec.values()]), )
        else:
            self._is_single_channel = True
            self.ob_ph_spec = self._basic_space_to_ph_spec(
                self.observation_space)
            self.flattened_ob_shape = self.ob_ph_spec[1][1:]

        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_ph_spec = (self.action_space.n, "Categorical")
        elif isinstance(self.action_space, gym.spaces.Box):
            self.action_ph_spec = (prod(self.action_space.shape),
                                   "DiagGaussian")
        else:
            raise ValueError("specified an unsupported action space {}".format(
                self.action_space))

    def _basic_space_to_ph_spec(self, sp):
        """Translate a gym space object to a tuple to specify data type and shape.
        Arguments:
            sp (obj): basic space object of gym interface.
        Returns:
            a tuple used for building TensorFlow placeholders where the first element specifies `dtype` and the second one specifies `shape`.
        """

        # (jones.wz) TO DO: handle gym Atari input
        if isinstance(sp, gym.spaces.Box):
            if len(sp.shape) == 3:
                return (tf.uint8, (None, ) + sp.shape)
            return (tf.float32, (None, prod(sp.shape)))
        elif isinstance(sp, gym.spaces.Discrete):
            return (tf.int32, (None, sp.n))
        elif isinstance(sp, gym.spaces.MultiDiscrete):
            return (tf.int32, (None, prod(sp.shape)))
        elif isinstance(sp, gym.spaces.MultiBinary):
            return (tf.int32, (None, prod(sp.shape)))
        else:
            raise TypeError(
                "specified an unsupported space type {}".format(sp))

    def flatten_obs(self, obs):
        """reshape the multi-channel observations into a flattern array for efficient communication in distributed training.

        Arguments:
            obs (obj): dict or list of numpy array for multi-channel observations.
        Returns:
            flattened_obs (tensor): a flattened array.
        """
        if isinstance(self.ob_ph_spec, list):
            assert len(obs) == len(
                self.observation_space
            ), "{} spaces for obs but {} inputs found".format(
                len(self.observation_space), len(obs))
            flattened_array = np.concatenate(
                [np.asarray(elm).astype(np.float32) for elm in obs], axis=1)
        elif isinstance(self.ob_ph_spec, OrderedDict):
            array_list = []
            for name in self.ob_ph_spec.keys():
                array_list.append(np.asarray(obs[name]).astype(np.float32))
            flattened_array = np.concatenate(array_list, axis=1)
        else:
            flattened_array = obs

        return flattened_array

    def reshape_flattened_obs(self, flattened_obs):
        """recovery the nested structure of flattened_obs

        Arguments:
            flattened_obs (obj): flattened array.
        Returns:
            the original nested struct of input obs.
        """
        tf2np_dtype = {
            tf.float32: np.float32,
            tf.float64: np.float64,
            tf.bool: np.bool,
            tf.int32: np.int32,
            tf.int64: np.int64,
            tf.int8: np.int8
        }

        if isinstance(self.ob_ph_spec, list):
            restore_obs = []
            cur_idx = 0
            for ph_dtype, ph_shape in self.ob_ph_spec:
                np_type = tf2np_dtype.get(ph_dtype, np.float32)
                restore_obs.append(
                    np.asarray(flattened_obs[:, cur_idx:cur_idx +
                                             ph_shape[1]]).astype(np_type))
                cur_idx += ph_shape[1]
        elif isinstance(self.ob_ph_spec, OrderedDict):
            restore_obs = {}
            cur_idx = 0
            for name, ph_tuple in self.ob_ph_spec.items():
                ph_dtype, ph_shape = ph_tuple
                np_type = tf2np_dtype.get(ph_dtype, np.float32)
                restore_obs[name] = np.asarray(
                    flattened_obs[:, cur_idx:cur_idx +
                                  ph_shape[1]]).astype(np_type)
        else:
            restore_obs = flattened_obs

        return restore_obs

    def feed_observations(self, placeholder, obs, train_size=None, offset=0):
        """Feed observations to their placeholders

        pair the data of each channel with their corresponding placeholder.
        Arguments:
            placeholder (obj): either a TF placeholder (ph), a list of phs, or a dict of phs.
            obs (obj): nested dict or list of numpy array of observations.
            train_size (int): size of train data for one optimization.
            offset (int): offset of current sub train data.
        Returns:
            feed_dict (dict): a dict mapping each ph to the corresponding numpy array.
        """

        feed_dict = dict()
        if isinstance(placeholder, list):
            assert len(placeholder) == len(
                obs), "{} placeholders but {} input supplied".format(
                    len(placeholder), len(obs))
            for ph, ob in zip(placeholder, obs):
                feed_dict[ph] = ob[offset:offset +
                                   train_size] if train_size else ob
        elif isinstance(placeholder, dict):
            assert len(placeholder) == len(
                obs), "{} placeholders but {} input supplied".format(
                    len(placeholder), len(self.observation_space))
            for ch_name in placeholder.keys():
                ph = placeholder[ch_name]
                feed_dict[ph] = obs[ch_name][
                    offset:offset + train_size] if train_size else obs[ch_name]
        else:
            feed_dict[placeholder] = obs[offset:offset +
                                         train_size] if train_size else obs

        return feed_dict

    def _restore_one_channel(self, sp, data, start_index):
        """Extract the data of one channel from the flattened data.

        Arguments:
            sp (obj): basic `gym.spaces.space` object.
            data (obj): a numpy array of flattened observations.
            start_index (int): indicating the starting index of this channel.
        Returns:
            selected_data (obj): a numpy array of this channel's data.
            dim (int): the dimensionality of this channel's data.
        """

        if isinstance(sp, gym.spaces.Box):
            dtype = np.float32
            dim = prod(sp.shape)
        elif isinstance(sp, gym.spaces.Discrete):
            dtype = np.int32
            dim = sp.n
        elif isinstance(sp, gym.spaces.MultiDiscrete):
            dtype = np.int32
            dim = prod(sp.shape)
        elif isinstance(sp, gym.spaces.MultiBinary):
            dtype = np.int32
            dim = prod(sp.shape)

        selected_data = np.asarray(
            data[:, start_index:start_index + dim]).astype(dtype)

        return selected_data, dim

    def setup(self,
              master,
              is_chief,
              global_step,
              ckpt_dir,
              summary_ops,
              global_vars=None,
              local_vars=None,
              save_var_list=None,
              save_steps=None,
              job_name="worker",
              task_index=0,
              async_mode=True):
        """
        Arguments:
            master (obj): specify the target of TF session.
            is_chief (bool): indicating whether this process is a chief worker.
            global_step (obj): the global_step var in the binded graph.
            ckpt_dir (str): specify the checkpoint directory of TF session.
            summary_ops (dict): a dict of TF summary operators.
            global_vars (list): global variables.
            local_vars (list): local variables.
            save_var_list (list): list of saveable variables.
            save_steps: (int): every save_steps to save checkpoint.
            export_dir (list): path to export SavedModel.
            job_name (str): job_name in distributed mode.
            task_index (int): task_index in distributed mode.
            async_mode (bool): indicating whether this is an asynchronous task.
        """

        if global_vars is not None:
            logger.info("in executor:")
            for v in global_vars:
                logger.info("{}".format(v))
            init_op = tf.variables_initializer(global_vars)
        else:
            # single-machine
            init_op = tf.global_variables_initializer()

        if local_vars is None:
            local_init_op = None
            ready_op = tf.report_uninitialized_variables(global_vars)
        else:
            pair_global_vars, pair_local_vars = self.get_variable_pairs(
                global_vars, local_vars)
            for gv, lv in zip(pair_global_vars, pair_local_vars):
                logger.info("{}, {}".format(gv, lv))
            local_init_op = tf.group(*([
                tf.assign(local_var, global_var) for local_var, global_var in
                zip(pair_local_vars, pair_global_vars)
            ]))
            ready_op = tf.report_uninitialized_variables(global_vars +
                                                         list(pair_local_vars))
        ready_for_local_init_op = tf.report_uninitialized_variables(
            global_vars)

        # create tensorflow saver object
        self.saver = tf.train.Saver(
            var_list=global_vars if save_var_list is None else save_var_list,
            reshape=False,
            sharded=False,
            max_to_keep=10,
            keep_checkpoint_every_n_hours=10000.0,
            name=None,
            restore_sequentially=False,
            saver_def=None,
            builder=None,
            defer_build=False,
            allow_empty=True,
            write_version=tf.train.SaverDef.V2,
            pad_step_number=False,
            save_relative_paths=True)

        # handle restore variables from checkpoint
        def init_fn(scaffold, session):
            if ckpt_dir:
                file = tf.train.latest_checkpoint(
                    checkpoint_dir=ckpt_dir, latest_filename=None)
                if file is not None:
                    logger.info('begin to restore model from {}'.format(file))
                    scaffold.saver.restore(sess=session, save_path=file)

        self.scaffold = tf.train.Scaffold(
            init_op=init_op,
            init_feed_dict=None,
            init_fn=init_fn,
            ready_op=ready_op,
            ready_for_local_init_op=ready_for_local_init_op,
            local_init_op=local_init_op,
            summary_op=None,
            saver=self.saver,
            copy_from_scaffold=None)

        self.do_summary = False
        for flag, summary_op_list in summary_ops.items():
            if len(summary_op_list) > 0:
                summary_ops[flag] = tf.summary.merge(summary_op_list)
            else:
                summary_ops[flag] = None
        if ckpt_dir:
            actor_summary_dir = os.path.join(ckpt_dir, "actor_summary")
            summary_dir = os.path.join(ckpt_dir, "worker_summary")
            summary_hook = easy_rl.utils.hooks.UpdateSummarySaverHook(
                self,
                global_step,
                job_name,
                task_index,
                save_steps=(save_steps or 100),
                output_dir=actor_summary_dir
                if job_name == "actor" else summary_dir,
                summary_op=summary_ops)
            saver_hook = tf.train.CheckpointSaverHook(
                checkpoint_dir=ckpt_dir,
                save_steps=(save_steps or 300),
                scaffold=self.scaffold,
                checkpoint_basename='model.ckpt')
            chief_only_hooks = [saver_hook]
            hooks = [summary_hook]
        else:
            chief_only_hooks = []
            hooks = []

        # filter devices for asynchronous training
        if async_mode:
            if job_name == "learner":
                device_filters = [
                    '/job:ps', '/job:memory',
                    '/job:{job_name}/task:{task_index}'.format(
                        job_name=job_name, task_index=task_index)
                ]
            else:
                device_filters = None
            config_proto = tf.ConfigProto(device_filters=device_filters)
        else:
            config_proto = None
        self.session = tf.train.MonitoredTrainingSession(
            master=master,
            is_chief=is_chief,
            checkpoint_dir=None,
            scaffold=self.scaffold,
            chief_only_hooks=chief_only_hooks,
            hooks=hooks,
            save_summaries_steps=None,
            save_summaries_secs=None,
            config=config_proto)

    def run(self, fetches, feed_dict, options=None, run_metadata=None):
        """Execution

        Run the given `fetches` with the given `feed_dict` by `self.session`.
        Arguments:
            fetches: a list of (nested) tensor-like objects.
            feed_dict (dict): keys are tensors (usually placeholders), values are numpy array.
        Returns: the fetched values of each op.
        """

        return self.session.run(
            fetches,
            feed_dict=feed_dict,
            options=options,
            run_metadata=run_metadata)

    def restore_action_shape_if_needed(self, actions):
        """
        Arguments:
            actions (obj): 2-dimensional numpy array of shape (batch_size, flattened_action_shape).
        Returns: the reshaped actions.
        """
        if isinstance(self.action_space, gym.spaces.Box):
            return np.reshape(
                actions, [actions.shape[0]] + list(self.action_space.shape))
        return actions

    def get_variable_pairs(self, global_vars, local_vars):
        local_global_pairs = []
        name_to_vars = {}
        for var in global_vars:
            name = var.name
            if re.search("/", name):
                m = re.match("([^/]*)(/.*)", name)
                scope_name = m.group(1)
                name = m.group(2)
                if scope_name in name:
                    name = name.lstrip('/' + scope_name)
                name_to_vars[name] = var
            else:
                name_to_vars[name] = var

        for var in local_vars:
            name = var.name
            if re.search("/", name):
                m = re.match("([^/]*)(/.*)", name)
                scope_name = m.group(1)
                name = m.group(2)
                if scope_name in name:
                    name = name.lstrip('/' + scope_name)

            if name in name_to_vars:
                local_global_pairs.append((name_to_vars[name], var))

        return zip(*local_global_pairs)

    def unsafe_unfinalize(self):
        self.session.graph._unsafe_unfinalize()

    def finalize(self):
        self.session.graph.finalize()
