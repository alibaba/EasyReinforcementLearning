# ==================================================
#
# Copyright (c) 2018, Alibaba Inc.
# All Rights Reserved.
#
# ==================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pickle, cloudpickle
from multiprocessing import Process, Pipe
from easy_rl.utils.gym_wrapper.atari_wrapper import make_atari, wrap_deepmind


def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn.data()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            state, reward, terminal, info = env.step(data)
            if terminal:
                state = env.reset()
            remote.send((state, reward, terminal, info))
        elif cmd == 'reset':
            state = env.reset()
            remote.send(state)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spec':
            remote.send((env.observation_space, env.action_space))
        else:
            raise ValueError("unkowned `{}` command!".format(cmd))


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents
    """

    def __init__(self, data):
        self.data = data

    def __getstate__(self):
        return cloudpickle.dumps(self.data)

    def __setstate__(self, ob):
        self.data = pickle.loads(ob)


class VectorizedEnvironment():
    def __init__(self, make_env, num_env, seed=0, **kwargs):
        """
        Vectorized environments to run in subprocesses.
        Args:
            environment: a dict of input single enviroment spec.
            num_env: batch environment size.
            seed: random seed.
        """

        env_funcs = [make_env(seed + i) for i in range(num_env)]
        self.waiting = False
        self.closed = False
        self.num_env = num_env
        self.batch_size = num_env
        self.remotes, self.work_remotes = zip(
            *[Pipe() for _ in range(self.num_env)])
        self.ps = [
            Process(
                target=worker,
                args=(work_remote, remote, CloudpickleWrapper(env_func)))
            for (work_remote, remote,
                 env_func) in zip(self.work_remotes, self.remotes, env_funcs)
        ]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spec', None))
        self.states_spec, self.actions_spec = self.remotes[0].recv()

    def __str__(self):
        return 'Vectorized Environment'

    def step(self, actions):
        if self.closed:
            raise ValueError(
                "`execute` operation is not allowed, all environments closed!")
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, info = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), info

    def reset(self):
        if self.closed:
            raise ValueError(
                "`close` operation is not allowed, all environments closed!")
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


if __name__ == '__main__':

    obs, next_obs, dones, rewards, actions = [], [], [], [], []

    def make_env(rank):
        def make_atari_env():
            env = make_atari("PongNoFrameskip-v4")
            env = wrap_deepmind(
                env=env,
                frame_stack=True,
                clip_rewards=False,
                episode_life=True,
                wrap_frame=True,
                frame_resize=42)

            return env

        return make_atari_env

    vec_env = VectorizedEnvironment(make_env, num_env=4)

    vec_obs = vec_env.reset()

    for i in range(10):
        action = np.random.randint(6, size=(4, ))
        vec_next_obs, vec_reward, vec_done, vec_info = vec_env.step(
            actions=action)
        obs.append(vec_obs)
        next_obs.append(vec_next_obs)
        dones.append(vec_done)
        rewards.append(vec_reward)
        actions.append(action)

    obs_ = np.swapaxes(np.asarray(obs), 0, 1)
    next_obs_ = np.swapaxes(np.asarray(next_obs), 0, 1)
    dones_ = np.swapaxes(np.asarray(dones), 0, 1)
    rewards_ = np.swapaxes(np.asarray(rewards), 0, 1)
    actions_ = np.swapaxes(np.asarray(actions), 0, 1)

    a = 0
