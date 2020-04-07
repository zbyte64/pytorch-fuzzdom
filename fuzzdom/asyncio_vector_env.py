import numpy as np
from copy import deepcopy
import asyncio

import traceback
from unittest.mock import patch
import itertools

from gym import logger
from gym.vector.vector_env import VectorEnv
from gym.vector.utils import concatenate, create_empty_array

from .process import PytorchProcessPoolExecutor


def get_env_fn(env, fn, with_executor=False):
    fns = []
    while env:
        if with_executor and hasattr(type(env), f"exec_{fn}"):
            fns.append(getattr(env, f"exec_{fn}"))
        elif hasattr(type(env), fn):
            fns.append(no_exec(getattr(env, fn)))
        env = getattr(env, "env", None)
    fns.reverse()
    return fns


def no_exec(f):
    async def ret(v, exec=None):
        return f(v)

    return ret


async def eval_env_fn(fns, value, *args):
    for fn in fns:
        value = await fn(value, *args)
    return value


async def resolve_env_fn(env, value, fn, executor=None):
    fns = get_env_fn(env, fn, with_executor=executor is not None)
    if executor:
        return await eval_env_fn(fns, value, executor)
    return await eval_env_fn(fns, value)


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
    action = await resolve_env_fn(env, action, "action")
    result = await root_env.async_step(action)
    _ = await resolve_env_fn(env, result, "step_result")
    reward = await resolve_env_fn(env, result[1], "reward")
    # intensive serialization gets tasked out
    # obs = await resolve_env_fn(env, result[0], "observation", executor)
    return (result[0], reward, *result[2:])


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
        self.executor = PytorchProcessPoolExecutor(3)

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
                traceback.print_exception(type(result), result, result.__traceback__)
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
        observations = await self.process_observations(observations)
        concatenate(observations, self.observations, self.single_observation_space)

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones),
            infos,
        )

    def close(self, terminate=False):
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

    async def process_observations(self, observations):
        # coroutines are resets
        for tries_left in range(2, 0, -1):
            obs = map(
                lambda x: x[0]
                if asyncio.iscoroutine(x[0])
                else resolve_env_fn(x[1], x[0], "observation", self.executor),
                zip(observations, self.envs),
            )
            from concurrent.futures.process import BrokenProcessPool
            try:
                p = await asyncio.gather(*obs)
            except BrokenProcessPool as e:
                print("Remote Cause:", getattr(e, "__cause__", ""))
                self.executor.shutdown()
                if tries_left:
                    self.executor = PytorchProcessPoolExecutor(self.num_envs)
                else:
                    raise
            return p
