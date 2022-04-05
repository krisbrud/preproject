from typing import Union
import torch
import gym
import numpy as np

from assisted_baselines.common.assistant import BaseAssistant

class MockAssistant(BaseAssistant):
    def __init__(self, n_actions):
        super().__init__(n_actions=n_actions)
        self.n_actions = n_actions
    
    def get_action(self, observation: torch.Tensor) -> torch.Tensor:
        # action = np.zeros(shape=(self.n_actions, ))
        # TODO: Go max speed towards goal
        position_errors = self._preprocess_observation(observation)

        action = (-1) * position_errors / np.linalg.norm(position_errors, ord=2)

        return action
    
    def reset(self) -> None:
        return super().reset()

    def _preprocess_observation(self, observation: np.ndarray) -> Union[np.ndarray, torch.Tensor]:
        return observation[:2] # Return only rel_pos