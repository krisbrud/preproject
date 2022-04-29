from collections import namedtuple
from typing import List, NamedTuple
from assisted_baselines.common.assistant import BaseAssistant
import gym.spaces
import torch as th
import numpy as np
from typing import Union

from dataclasses import dataclass


# Code borrowed from https://github.com/simentha/gym-auv/blob/master/gym_auv/utils/controllers.py
# and modified
class PIDController:
    """
    PID controller with support for anti-wind-up. 
    To deactivate anti-wind-up, set valid_input_range to None.
    """
    def __init__(self, Kp, Ki, Kd, timestep, valid_input_range=[-1.0, 1.0]):
        assert timestep > 0.0, f"timestep {timestep} is not a positive number!"
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.timestep = timestep
        if valid_input_range is not None:
            assert len(valid_input_range) == 2
            self.min_input, self.max_input = valid_input_range
        else:
            self.min_input, self.max_input = float("-inf"), float("inf")
        self.reset()
        
    def reset(self) -> None:
        self._u = 0
        self.accumulated_error = 0
        self.last_error = 0
        
    def u(self, error):
        if self._u < self.min_input or self.max_input < self._u: 
            # anti wind up, don't accumulate the error
            pass
        else: 
            self.accumulated_error += error * self.timestep

        derivative_error = (error-self.last_error) / self.timestep
        self._u = self.Kp * error + self.Ki * self.accumulated_error \
                + self.Kd * derivative_error
        return self._u
    
class PIController(PIDController):
    def __init__(self, Kp, Ki, timestep):
        # Inherit from PID, set Kd to 0
        super(PIController, self).__init__(Kp=Kp, Ki=Ki, Kd=0, timestep=timestep)

class PIDGains(NamedTuple):
    Kp: float
    Ki: float
    Kd: float

class PIDAssistant(BaseAssistant):
    """
    :param observation_space
    :param action_space
    :param pid_controllers: list of PID controllers to assist training
    :param pid_error_action_map: list of tuples telling which observation to use for PID controller
    """
    def __init__(self,
                 n_actions: int,
                 pid_controllers: List[PIDController],
                 pid_error_indices: List[int]):
        super(PIDAssistant, self).__init__(n_actions=n_actions)
        
        self.pid_controllers = pid_controllers
        self.pid_error_indices = pid_error_indices


    # TODO: Make observation preprocessing
    def _preprocess_observation(self, observation: np.ndarray) -> Union[np.ndarray, th.Tensor]:
        relevant_observations = list(map(lambda idx: observation[idx], self.pid_error_indices))
        # Scale the observations
        obs = relevant_observations * np.array([2.0, np.pi, np.pi])
        return obs # relevant_observations

    def reset(self) -> None:
        for pid_controller in self.pid_controllers:
            pid_controller.reset()

    def get_action(self, observation) -> np.ndarray:
        preprocessed_obs = self._preprocess_observation(observation)
        # print("(obs, preprocessed)", observation, preprocessed_obs)
        assistant_actions = np.zeros(shape=(self.n_actions,), dtype=np.float32)
        
        for i, (obs, pid) in enumerate(zip(preprocessed_obs, self.pid_controllers)):
            assistant_actions[i] = pid.u(obs)

        return assistant_actions
            