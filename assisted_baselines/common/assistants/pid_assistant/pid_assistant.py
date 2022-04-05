from typing import List
from assisted_baselines.common.assistants.pid_assistant.pid_controller import PID
from assisted_baselines.common.assistant import BaseAssistant
import gym.spaces
import torch as th
import numpy as np
from typing import Union

class PIDAssistant(BaseAssistant):
    """
    :param observation_space
    :param action_space
    :param pid_controllers: list of PID controllers to assist training
    :param pid_error_action_map: list of tuples telling which observation to use for PID controller
    """
    def __init__(self,
                 n_actions: int,
                 pid_controllers: List[PID],
                 pid_error_indices: List[int],
                 use_noise=True,
                 noise_std = 0.1):
        super(PIDAssistant, self).__init__(n_actions=n_actions)
        
        self.pid_controllers = pid_controllers
        self.pid_error_indices = pid_error_indices
        self.use_noise = use_noise
        self.noise_std = noise_std


    # TODO: Make observation preprocessing
    def _preprocess_observation(self, observation: np.ndarray) -> Union[np.ndarray, th.Tensor]:
        relevant_observations = map(lambda idx: (-1) * observation[idx], self.pid_error_indices)
        return relevant_observations

    def reset(self) -> None:
        for pid_controller in self.pid_controllers:
            pid_controller.reset()

    def get_action(self, observation) -> th.Tensor:
        preprocessed_obs = self._preprocess_observation(observation)

        assistant_actions = th.zeros(size=(self.n_actions,))
        
        for i, (obs, pid) in enumerate(zip(preprocessed_obs, self.pid_controllers)):
            assistant_actions[i] = pid.u(obs)
            
        if self.use_noise:
            assistant_actions += th.randn_like(assistant_actions) * self.noise_std
        
        return assistant_actions
            