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

from collections import Iterable

import numpy as np
import scipy.signal
import tensorflow as tf


def discount(x, gamma, dones):
    """Compute the discounted cumulative sum

    e.g., when x=[1, 1, 1, 1, 1, 1],
    dones=[False, False, False, True, False, True] and gamma=0.9,
    the returned value is [3.439, 2.71, 1.9, 1, 1.9, 1]
    """
    acc = x[-1]
    ys = np.zeros(shape=np.shape(dones), dtype=np.float32)
    ys[-1] = acc
    for i in reversed(range(len(x) - 1)):
        y = x[i] + gamma * acc * (1 - dones[i])
        acc = y
        ys[i] = y
    return ys


def unflatten_tensor(flatten_vars, var_shapes_list):
    """Change a flattened tensor back

    Arguments:
        flatten_vars (tensor): flattened variables.
        var_shapes_list (list): the shape of each variable.

    Returns:
        outputs (list): a list of the recovered variables.
    """
    outputs = []
    offset = 0
    for shape_ in var_shapes_list:
        params_size = prod(shape_)
        outputs.append(
            tf.reshape(flatten_vars[offset:offset + params_size], shape_))
        offset += params_size
    return outputs


def prod(xs):
    """Computes the product along the elements in an iterable. Returns 1 for empty iterable.

    Arguments:
        xs (obj): integer or iterable containing numbers.
    Returns:
        p (int): product along each axis.
    """
    if not isinstance(xs, Iterable):
        return 1
    p = 1
    for x in xs:
        p *= x
    return p


def n_step_adjustment(obs,
                      actions,
                      rewards,
                      new_obs,
                      dones,
                      gamma=0.99,
                      n_step=1):
    """Adjust rewards to make TD(n)

    reward[i] = (
        reward[i] * gamma**0 +
        reward[i+1] * gamma**1 +
        ... +
    reward[i+n_step-1] * gamma**(n_step-1))

    The ith new_obs is also adjusted to point to the (i+n_step-1)'th new obs.
    """

    traj_length = len(rewards) - n_step + 1
    for i in range(traj_length):
        if dones[i]:
            continue
        for j in range(1, n_step):
            new_obs[i] = new_obs[i + j]
            dones[i] = dones[i + j]
            rewards[i] += gamma**j * rewards[i + j]
            if dones[i]:
                break
    return np.asarray(obs[:traj_length]), np.asarray(
        actions[:traj_length]), np.asarray(rewards[:traj_length]), np.asarray(
            new_obs[:traj_length]), np.asarray(dones[:traj_length])


def compute_targets(rewards,
                    dones,
                    value_preds,
                    gamma=0.9,
                    lambda_=0.5,
                    use_gae=True):
    """Compute the advantages of sampled state-action pairs as well as the state value targets to be approximated.

    Note that the last transition may be dropped when the last element of value_preds is used.

    Arguments:
        rewards, dones are basic fields generated via interactions.
        values (obj): estimated state values.
        gamma (float): discount factor.
        lambda_ (float): the GAE factor.
    Returns:
        advantages (obj): advantages of the collected state-action pairs.
        value_targets (obj): targets of the state value function approximator
    """

    if use_gae:
        assert value_preds is not None, "GAE requires predicted state values"
        value_preds = np.asarray(value_preds, dtype=np.float32)
        delta_t = rewards[:-1] + gamma * value_preds[1:] * (
            1 - np.asarray(dones[:-1])) - value_preds[:-1]
        # This formula for the advantage comes
        # "Generalized Advantage Estimation": https://arxiv.org/abs/1506.02438
        advantages = discount(delta_t, gamma * lambda_, dones[:-1])
        value_targets = (advantages + value_preds[:-1]).copy().astype(
            np.float32)
    else:
        if value_preds is None:
            returns = discount(rewards, gamma, dones[:-1])
            advantages = returns.copy().astype(np.float32) - 0
            value_targets = returns
        else:
            value_preds = np.asarray(value_preds, dtype=np.float32)
            last_r = value_preds[-1]
            rewards_plus_v = np.concatenate([rewards[:-1], np.array([last_r])])
            returns = discount(rewards_plus_v, gamma, dones[:-1])
            advantages = returns.copy().astype(np.float32) - value_preds[:-1]
            value_targets = returns

    return advantages, value_targets
