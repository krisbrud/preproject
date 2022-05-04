from typing import Union
from assisted_baselines.common.assistant import BaseAssistant
import numpy as np
import torch


class MountainCarAssistant(BaseAssistant):
    def __init__(self, n_actions=1, heuristic_action=0.7):
        super().__init__(n_actions)

        # Define high and low
        x_bottom = -(np.pi / 2) / 3  # Bottom of mountain valley
        x_delta = 0.3
        self._low = x_bottom - x_delta
        self._high = x_bottom + x_delta

        self.heuristic_action = heuristic_action

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        # Return an action of size heuristic action in the direction of
        # the velocity
        pos, vel = observation
        is_in_heuristic_zone = self._low <= pos <= self._high
        if not is_in_heuristic_zone:
            # Don't apply force/torque
            return 0

        # If in heuristic zone, give action
        if vel >= 0:
            return self.heuristic_action
        else:
            return -self.heuristic_action

    def _preprocess_observation(
        self, observation: np.ndarray
    ) -> Union[np.ndarray, torch.Tensor]:
        return super()._preprocess_observation(observation)

    def reset(self) -> None:
        pass
