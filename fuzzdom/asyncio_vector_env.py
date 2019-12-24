import numpy as np
from copy import deepcopy
import threading
import time
import asyncio
from queue import Queue, Empty
import traceback
from unittest.mock import patch

from gym import logger
from gym.vector.vector_env import VectorEnv
from gym.vector.utils import concatenate, create_empty_array


def resolve_env_fn(env, value, fn):
    if getattr(env, "env", None):
        value = resolve_env_fn(env.env, value, fn)
    if hasattr(env, fn):
        if hasattr(env, "env") and getattr(env.env, fn, None) == getattr(env, fn):
            return value
        value = getattr(env, fn)(value)
    return value


async def call_wrapped_async_reset(env, *args):
    root_env = env
    while hasattr(root_env, "env"):
        root_env = root_env.env
    result = await root_env.async_reset(*args)
    if env != root_env:
        with patch.object(root_env, "reset", return_value=result) as mock_method:
            result = env.reset(*args)
    return result


async def call_wrapped_async_step(env, action):
    root_env = env
    while hasattr(root_env, "env"):
        root_env = root_env.env
    action = resolve_env_fn(env, action, "action")
    result = await root_env.async_step(action)
    _ = resolve_env_fn(env, result, "step_result")
    reward = resolve_env_fn(env, result[1], "reward")
    obs = resolve_env_fn(env, result[0], "observation")
    return (obs, reward, *result[2:])


class AsyncioVectorEnv(VectorEnv):
    """Vectorized environment that runs multiple evented environments.

    Parameters
    ----------
    envs : iterable of environments.

    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.

    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.

    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.
    """

    def __init__(self, envs, observation_space=None, action_space=None, copy=True):
        self.envs = envs
        self.copy = copy

        if (observation_space is None) or (action_space is None):
            _env = self.envs[0]
            observation_space = observation_space or _env.observation_space
            action_space = action_space or _env.action_space
        super(AsyncioVectorEnv, self).__init__(
            num_envs=len(self.envs),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_observation_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self._actions = None
        self.closed = False
        self.loop = asyncio.get_event_loop()

    def seed(self, seeds=None):
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        self.loop.run_until_complete(
            asyncio.gather(*[env.seed(seed) for env, seed in zip(self.envs, seeds)])
        )

    def reset_async(self):
        pass

    def reset_wait(self):
        return self.loop.run_until_complete(self.async_reset())

    async def async_reset(self):
        self._dones[:] = False
        observations = await (
            asyncio.gather(*[call_wrapped_async_reset(env) for env in self.envs])
        )

        try:
            concatenate(observations, self.observations, self.single_observation_space)
        except ValueError:
            print([n.shape for n in observations])
            raise

        return np.copy(self.observations) if self.copy else self.observations

    def step_async(self, actions):
        self._original_actions = actions

    def step_wait(self):
        return self.loop.run_until_complete(self.async_step(self._original_actions))

    async def async_step(self, actions):
        observations, infos = [], []
        p = await (
            asyncio.gather(
                *[
                    asyncio.wait_for(call_wrapped_async_step(env, action), timeout=1.0)
                    for env, action in zip(self.envs, actions)
                ],
                return_exceptions=True,
            )
        )
        for i, result in enumerate(p):
            env = self.envs[i]
            if isinstance(result, Exception):
                print("Error:", type(result), result, self.envs[i].task)
                # traceback.print_exception(result)
                observation, self._rewards[i], self._dones[i], info = [
                    call_wrapped_async_reset(env),
                    0.0,
                    False,
                    {"bad_transition": True},
                ]
            else:
                observation, self._rewards[i], self._dones[i], info = result
            observations.append(observation)
            infos.append(info)
        p = await (asyncio.gather(*[o for o in observations if asyncio.iscoroutine(o)]))
        j = 0
        for i, o in enumerate(observations):
            if asyncio.iscoroutine(o):
                observations[i] = p[j]
                j += 1

        concatenate(observations, self.observations, self.single_observation_space)

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones),
            infos,
        )

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.loop.run_until_complete(
            asyncio.gather(
                *[asyncio.wait_for(env.close(), 1.0) for env in self.envs],
                return_exceptions=True,
            )
        )
        self.closed = True

    def _check_observation_spaces(self):
        pass
