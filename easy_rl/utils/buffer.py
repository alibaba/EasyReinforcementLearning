from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import sys
import time

from easy_rl.utils.segment_tree import SumSegmentTree, MinSegmentTree
from easy_rl.utils.window_stat import WindowStat


class ReplayBuffer(object):
    """Basic replay buffer.

    Support O(1) `add` and O(1) `sample` operations (w.r.t. each transition).
    The buffer is implemented as a fixed-length list where the index of insertion is reset to zero,
    once the list length is reached.
    """

    def __init__(self, size):
        """Create the replay buffer.

        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        """
        self._maxsize = size
        self._next_idx = 0
        self._hit_count = np.zeros(size)
        self._eviction_started = False
        self._num_added = 0
        self._num_sampled = 0
        self._evicted_hit_stats = WindowStat("evicted_hit", 1000)
        self._est_size_bytes = 0

        self._extra_fields = None

        self._first_add = True

    def __len__(self):
        return min(self._num_added, self._maxsize)

    def add(self,
            obs,
            actions,
            rewards,
            dones,
            next_obs=None,
            weights=None,
            **kwargs):

        batch_size = np.shape(rewards)[0]
        assert batch_size < self._maxsize, "size of data added in buffer is too big at once"

        truncated_size = min(batch_size, self._maxsize - self._next_idx)
        extra_size = max(0, batch_size - (self._maxsize - self._next_idx))

        if self._extra_fields is None:
            self._extra_fields = list(kwargs.keys())

        if self._first_add:
            self._obs = np.zeros(
                shape=((self._maxsize, ) + np.shape(obs)[1:]), dtype=obs.dtype)
            self._actions = np.zeros(
                shape=((self._maxsize, ) + np.shape(actions)[1:]),
                dtype=actions.dtype)
            self._rewards = np.zeros(shape=(self._maxsize, ), dtype=np.float32)

            if next_obs is not None:
                self._next_obs = np.zeros(
                    shape=((self._maxsize, ) + np.shape(next_obs)[1:]),
                    dtype=next_obs.dtype)

            if weights is not None:
                self._weights = np.zeros(
                    shape=((self._maxsize, )), dtype=np.float32)

            self._dones = np.zeros(shape=(self._maxsize, ), dtype=np.float32)

            self._extras = {
                name: np.zeros(
                    shape=((self._maxsize, ) + np.shape(kwargs[name])[1:]),
                    dtype=kwargs[name].dtype)
                for name in self._extra_fields
            }

            self._first_add = False

        self._num_added += batch_size

        #if self._num_added <= self._maxsize:
        #self._est_size_bytes += sum(sys.getsizeof(d) for d in data)

        self._obs[self._next_idx:self._next_idx +
                  truncated_size] = obs[:truncated_size]
        self._actions[self._next_idx:self._next_idx +
                      truncated_size] = actions[:truncated_size]
        self._rewards[self._next_idx:self._next_idx +
                      truncated_size] = rewards[:truncated_size]
        self._dones[self._next_idx:self._next_idx +
                    truncated_size] = dones[:truncated_size]

        if next_obs is not None:
            self._next_obs[self._next_idx:self._next_idx +
                           truncated_size] = next_obs[:truncated_size]
        if weights is not None:
            self._weights[self._next_idx:self._next_idx +
                          truncated_size] = weights[:truncated_size]

        for name in self._extras.keys():
            self._extras[name][self._next_idx:self._next_idx +
                               truncated_size] = kwargs[name][:truncated_size]

        if extra_size > 0:
            self._obs[:extra_size] = obs[truncated_size:]
            self._actions[:extra_size] = actions[truncated_size:]
            self._rewards[:extra_size] = rewards[truncated_size:]
            self._dones[:extra_size] = dones[truncated_size:]
            if next_obs is not None:
                self._next_obs[:extra_size] = next_obs[truncated_size:]
            if weights is not None:
                self._weights[:extra_size] = weights[truncated_size:]

            for name in self._extras.keys():
                self._extras[name][:extra_size] = kwargs[name][truncated_size:]

        if self._next_idx + batch_size >= self._maxsize:
            self._eviction_started = True
        self._cover_indices = [
            self._next_idx + i for i in range(truncated_size)
        ]
        if extra_size > 0:
            self._cover_indices += [i for i in range(extra_size)]
        self._next_idx = (self._next_idx + batch_size) % self._maxsize
        if self._eviction_started:
            for i in self._cover_indices:
                self._evicted_hit_stats.push(self._hit_count[i])
                self._hit_count[i] = 0

    def _encode_sample(self, idxes):
        idxes = np.asarray(idxes)

        obs = np.take(self._obs, indices=idxes, axis=0)
        actions = np.take(self._actions, indices=idxes, axis=0)
        rewards = np.take(self._rewards, indices=idxes, axis=0)
        next_obs = np.take(self._next_obs, indices=idxes, axis=0)
        dones = np.take(self._dones, indices=idxes, axis=0)

        batch_data = dict(
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_obs=next_obs)

        return batch_data

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
          batch of observations
        act_batch: np.array
          batch of actions executed given obs_batch
        rew_batch: np.array
          rewards received as results of executing act_batch
        next_obs_batch: np.array
          next set of observations seen after executing act_batch
        done_mask: np.array
          done_mask[i] = 1 if executing act_batch[i] resulted in
          the end of an episode and 0 otherwise.
        """
        idxes = np.random.randint(
            0, min(self._num_added, self._maxsize) - 1, size=(batch_size, ))
        self._num_sampled += batch_size
        return self._encode_sample(idxes)

    def stats(self, debug=False):
        data = {
            "added_count": self._num_added,
            "sampled_count": self._num_sampled,
            "est_size_bytes": self._est_size_bytes,
            "num_entries": len(self),
        }
        if debug:
            data.update(self._evicted_hit_stats.stats())
        return data


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
          Max number of transitions to store in the buffer. When the buffer
          overflows the old memories are dropped.
        alpha: float
          how much prioritization is used
          (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._prio_change_stats = WindowStat("reprio", 1000)

        self._debug_cost = 0

    def add(self, obs, actions, rewards, dones, next_obs, weights, **kwargs):
        """See ReplayBuffer.store_effect"""

        super(PrioritizedReplayBuffer, self).add(
            obs=obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
            next_obs=next_obs,
            **{})

        if weights is None:
            weights = self._max_priority
            constant_weight = weights**self._alpha
            for idx in self._cover_indices:
                self._it_sum[idx] = constant_weight
                self._it_min[idx] = constant_weight
        else:
            weights = np.power(weights, self._alpha)
            for n, idx in enumerate(self._cover_indices):
                self._it_sum[idx] = weights[n]
                self._it_min[idx] = weights[n]

    def _sample_proportional(self, batch_size):
        res = []
        sum_value = self._it_sum.sum(0, len(self))
        mass = np.random.random(size=batch_size) * sum_value
        for i in range(batch_size):
            # TODO(szymon): should we ensure no repeats?
            idx = self._it_sum.find_prefixsum_idx(mass[i])
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        """Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.


        Parameters
        ----------
        batch_size: int
          How many transitions to sample.
        beta: float
          To what degree to use importance weights
          (0 - no corrections, 1 - full correction)

        Returns
        -------
        obs_batch: np.array
          batch of observations
        act_batch: np.array
          batch of actions executed given obs_batch
        rew_batch: np.array
          rewards received as results of executing act_batch
        next_obs_batch: np.array
          next set of observations seen after executing act_batch
        done_mask: np.array
          done_mask[i] = 1 if executing act_batch[i] resulted in
          the end of an episode and 0 otherwise.
        weights: np.array
          Array of shape (batch_size,) and dtype np.float32
          denoting importance weight of each sampled transition
        idxes: np.array
          Array of shape (batch_size,) and dtype np.int32
          idexes in buffer of sampled experiences
        """
        assert beta > 0
        self._num_sampled += batch_size

        start = time.time()
        idxes = self._sample_proportional(batch_size)
        self._debug_cost += time.time() - start

        sum_value = self._it_sum.sum()

        weights = []
        p_min = self._it_min.min() / sum_value
        max_weight = (p_min * len(self))**(-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / sum_value
            weight = (p_sample * len(self))**(-beta)
            weights.append(weight / max_weight)
        weights = np.asarray(weights)
        encoded_sample = self._encode_sample(idxes)
        encoded_sample["weights"] = weights
        encoded_sample["indexes"] = idxes
        return encoded_sample

    def update_priorities(self, indexes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        indexes: [int]
          List of idxes of sampled transitions
        priorities: [float]
          List of updated priorities corresponding to
          transitions at the sampled idxes denoted by
          variable `idxes`.
        """
        assert len(indexes) == len(priorities)
        pvs = np.power(priorities, self._alpha).astype(np.float64)
        for idx, priority, pv in zip(indexes, priorities, pvs):
            assert priority > 0
            assert 0 <= idx < len(self)
            delta = pv - self._it_sum[idx]
            self._prio_change_stats.push(delta)
            self._it_sum[idx] = pv
            self._it_min[idx] = pv

        self._max_priority = max(self._max_priority, np.max(priorities))

    def stats(self, debug=False):
        parent = ReplayBuffer.stats(self, debug)
        if debug:
            parent.update(self._prio_change_stats.stats())
        return parent


class TrajectoryBuffer(ReplayBuffer):
    """Basic trajectory buffer.

    Transitions are stored in chronological order

    The behavior of TrajectoryBuffer is like a map-reduce operation,
    add corresponds to map, sample corresponds to reduce.
    Once sample data from TrajectoryBuffer, the current number of samples and will be reset to 0
    regardless of the actual sample size.
    """

    def __init__(self, size=10000):
        super(TrajectoryBuffer, self).__init__(size)

        self._extra_fields = None
        self._current_num = 0

    def __len__(self):
        return self._current_num

    def add(self, obs, actions, rewards, dones, **kwargs):

        super(TrajectoryBuffer, self).add(
            obs=obs, actions=actions, rewards=rewards, dones=dones, **kwargs)

        batch_size = np.shape(rewards)[0]
        self._current_num += batch_size
        self._current_num = min(self._current_num, self._maxsize)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Arguments:
            batch_size (int): How many transitions to sample.
        Returns:
            trajectories (obj): a batch of `batch_size` transitions where each consists of several fields for policy optimization methods.
        """
        sample_size = min(batch_size, len(self))

        obs = self._obs[:sample_size]
        actions = self._actions[:sample_size]
        rewards = self._rewards[:sample_size]
        dones = self._dones[:sample_size]

        extra_data = {
            name: v[:sample_size]
            for name, v in self._extras.items()
        }

        self._num_sampled += sample_size

        # reset the state of buffer
        self._current_num = 0
        self._next_idx = 0

        batch_data = dict(
            obs=np.asarray(obs),
            actions=np.asarray(actions),
            rewards=np.asarray(rewards),
            dones=np.asarray(dones))
        for key, v in extra_data.items():
            batch_data[key] = v
        return batch_data

    def stats(self, debug=False):
        data = {
            "added_count": self._num_added,
            "sampled_count": self._num_sampled,
        }
        return data


class AggregateBuffer(object):
    """Aggregate buffer.

    A centre storage to aggregate the samples from actors and
    send out all the collection to learner at once.

    """

    def __init__(self, size):
        self._maxsize = size
        self._next_idx = 0
        self._num_added = 0
        self._num_sampled = 0
        self._fields = None
        self._first_add = True

    def __len__(self):
        return min(self._num_added, self._maxsize)

    def add(self, **kwargs):
        """
        data passed in dict will be stored.
        """
        if self._fields is None:
            self._fields = list(sorted(kwargs.keys()))

        if self._first_add:
            self._values = {
                name: np.zeros(
                    shape=((self._maxsize, ) + np.shape(kwargs[name])[1:]),
                    dtype=np.float32)
                for name in self._fields
            }

            self._first_add = False

        batch_size = np.shape(next(iter(kwargs.values())))[0]
        truncated_size = min(batch_size, self._maxsize - self._next_idx)
        extra_size = max(0, batch_size - (self._maxsize - self._next_idx))

        self._num_added += batch_size

        for name in self._values.keys():
            self._values[name][self._next_idx:self._next_idx +
                               truncated_size] = kwargs[name][:truncated_size]

        if extra_size > 0:
            for name in self._values.keys():
                self._values[name][:extra_size] = kwargs[name][truncated_size:]

        self._cover_indices = [
            self._next_idx + i for i in range(truncated_size)
        ]
        if extra_size > 0:
            self._cover_indices += [i for i in range(extra_size)]
        self._next_idx = (self._next_idx + batch_size) % self._maxsize

    def sample(self, batch_size):
        sample_size = min(batch_size, len(self))

        batch_data = {
            name: self._values[name][:sample_size]
            for name in self._fields
        }

        self._num_sampled += sample_size

        # reset the state of buffer
        self._current_num = 0
        self._next_idx = 0

        return batch_data
