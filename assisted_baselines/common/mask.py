from abc import ABC, abstractmethod
from argparse import ArgumentError
from typing import Union

import gym.spaces
import stable_baselines3
from stable_baselines3.common.distributions import DiagGaussianDistribution
import torch
import numpy as np

AGENT_FLAG = True
ASSISTANT_FLAG = False


class ActiveActionsMask:
    """
    Mask that may be used to choose which actions are to be decided by the agent and which by
    the assistant in a continuous multi-input action space.
    Agent actions where mask is True, and assistant actions where mask is False.

    :param action_space - The gym environment's action space
    :param action_mask - A numpy array with
    """

    def __init__(self, n_actions: int, mask: Union[torch.Tensor, np.ndarray]) -> None:
        self._n_actions = n_actions

        if isinstance(mask, np.ndarray):
            self._mask = torch.Tensor(mask.astype(bool)).type(torch.BoolTensor)
        elif isinstance(mask, torch.Tensor):
            self._mask = mask
        else:
            raise ArgumentError(
                f"Expected mask to be numpy ndarray or torch Tensor, but got {type(mask)}!"
            )

        if self._mask.shape != (self._n_actions,):
            raise ValueError(
                f"The shape of the action space: {self._action_space.shape} did not match the shape of the mask: {mask.shape}!"
            )

    def mix_np(self, agent_actions: np.ndarray, assistant_actions: np.ndarray):
        """ """
        # Convert to torch and use torch method
        # print("mix_np agent actions", agent_actions)
        # print("mix_np assistant actions", assistant_actions)

        agent_action_tensor = torch.from_numpy(agent_actions)
        assistant_action_tensor = torch.from_numpy(assistant_actions)

        masked_action_tensor = self.mix(
            agent_actions=agent_action_tensor, assistant_actions=assistant_action_tensor
        )

        return masked_action_tensor.cpu().detach().numpy()

    def mix(self, agent_actions: torch.Tensor, assistant_actions: torch.Tensor):
        """
        Applies the mask, returns the agent_actions where the mask is True
        and the assitant_actions where the mask is False
        """
        if agent_actions.shape[-1] != self._n_actions:
            raise ValueError(
                f"Shape of agent_actions: {agent_actions.shape}"
                f"does not match number of actions {self._n_actions}!"
            )
        if assistant_actions.shape[-1] != self._n_actions:
            raise ValueError(
                f"Shape of assistant_actions: {agent_actions.shape}"
                f"does not match number of actions {self._n_actions}!"
            )

        actions = torch.where(self.mask, agent_actions, assistant_actions)

        return actions

    def select(self, tensor: torch.Tensor):
        """
        TODO Documentation
        Returns the parts of the samples where the mask is True
        """
        if len(tensor.shape) > 1:
            return tensor[:, self.mask]
        else:
            return tensor[self.mask]

    @property
    def mask(self):
        """
        Getter for the mask
        """
        return self._mask


# The MaskSchedule (which should inherit from this base) is used to decide which ActiveActionMask is used during training.
# It is called once before each rollout.
# It's argument is the remaining progress, which decreases linearly from 1.0 at the start of training to 0.0 at the end of training.
class BaseMaskSchedule(ABC):
    def __init__(self):
        super(BaseMaskSchedule, self).__init__()

    @abstractmethod
    def get_mask(timestep: int) -> ActiveActionsMask:
        pass
