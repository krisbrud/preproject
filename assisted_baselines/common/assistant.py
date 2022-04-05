from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import gym.spaces
import gym.core
import torch
import numpy as np

from assisted_baselines.common.mask import ActiveActionsMask

class BaseAssistant(ABC):
    def __init__(self, n_actions):
        self.n_actions = n_actions

    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        The get_action method should return the actions suggested by the assistant. 
        If an assistant is only implemented for some of the actions, return zeros for the other
        actions and make sure the action is not used through the ActiveActionMask.
        """
        pass

    @abstractmethod
    def _preprocess_observation(self, observation: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        """
        Preprocesses the observation so that it may be given in the format required by the assistant
        """
        return observation

    @abstractmethod
    def reset(self) -> None:
        """
        Resets the state of the assistant. This allows for recurrent controllers with state
        tied to an individual environment, e.g. as used in SubProcVecEnv
        """
        pass


class AssistantWrapper(gym.core.Wrapper):
    r"""Applies the actions predicted by the assistant according to the current mask.

    Also allows for the owner of the object to update the mask.
    """

    def __init__(self, env, assistant: Union[BaseAssistant, None]=None):
        super().__init__(env)

        if assistant is None:
            raise ValueError("AssistantWrapper called without assistant!")
        
        self.assistant = assistant

        self._assistant_action = None

    def observation(self, observation):
        # No action needed
        return observation

    def get_assistant_action(self):
        """
        Gets the assistants predicted action.
        """
        # Avoid evaluating the assistant action more than once by caching 
        # the assistant action every step.
        # As the assistant may be recurrent (and thus stateful), it 
        # should only be called once per step.
        if self._assistant_action is not None:
            return self._assistant_action

        obs = self.env.observe()
        self._assistant_action = self.assistant.get_action(observation=obs)

        return self._assistant_action

    def step(self, action):
        self._assistant_action = None # Reset assistant action caching

        return super().step(action)

    def reset(self, **kwargs):
        self.assistant.reset()

        self._assistant_action = None # Reset assistant action caching
        
        return super().reset(**kwargs)