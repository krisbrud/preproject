
from typing import NamedTuple, Optional, Union

import numpy as np
import torch as th
from gym import spaces

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.buffers import BaseBuffer

class ValueBufferSamples(NamedTuple):
    observations: th.Tensor
    value_targets: th.Tensor
class ValueReplayBuffer(BaseBuffer):
    """
    Replay buffer that stores states and value targets for use in PPG (Phasic Policy Gradient).

    Most code borrowed from ReplayBuffer in sb3, with unused code removed.
    https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device:
    :param n_envs: Number of parallel environments
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "cpu",
        n_envs: int = 1,
    ):
        super(ValueReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        self.observations = np.zeros((self.buffer_size, self.n_envs) + self.obs_shape, dtype=observation_space.dtype)
        self.value_targets = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        
    def add(
        self,
        obs: np.ndarray,
        value_target: np.ndarray
    ) -> None:

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs,) + self.obs_shape)
            next_obs = next_obs.reshape((self.n_envs,) + self.obs_shape)

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()
        self.value_targets[self.pos] = np.array(value_target).copy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, env: Optional[VecNormalize] = None) -> ValueBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super().sample(batch_size=batch_size, env=env)

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ValueBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.value_targets[batch_inds, env_indices]
        )
        return ValueBufferSamples(*tuple(map(self.to_torch, data)))