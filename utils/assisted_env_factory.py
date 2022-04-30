from typing import Union
import gym
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from assisted_baselines.common.assistant import AssistantWrapper

from config import Config
from assisted_baselines.common.assistants.pid import (
    PIDAssistant,
    PIDController,
    PIDGains,
)


def make_pid_controller(gains: PIDGains, timestep: float):
    pid_controller = PIDController(
        Kp=gains.Kp, Ki=gains.Ki, Kd=gains.Kd, timestep=timestep
    )

    return pid_controller


def make_pid_assistant(cfg: Config):
    timestep = 0.1  # cfg.timestep
    auv_pid_gains = cfg.assistance.auv_pid
    # surge_gains = get_gains(auv_pid_gains.surge)
    surge_pid = make_pid_controller(auv_pid_gains.surge, timestep)

    # rudder_gains = get_gains(auv_pid_gains.rudder)
    rudder_pid = make_pid_controller(auv_pid_gains.rudder, timestep)

    # elevator_gains = get_gains(auv_pid_gains.elevator)
    elevator_pid = make_pid_controller(auv_pid_gains.elevator, timestep)

    pid_assistant = PIDAssistant(
        pid_controllers=[surge_pid, rudder_pid, elevator_pid],
        pid_error_indices=[9, 10, 11],
        n_actions=3,
    )
    return pid_assistant


def get_eval_env(cfg: Config):
    env = AssistantWrapper(
        gym.make(cfg.env.name),
        make_pid_assistant(cfg),
        initial_mask=cfg.assistance.mask_schedule.get_mask(0),
    )
    return env


def get_assisted_envs(cfg: Config):
    if cfg.train.num_envs > 1:
        env = SubprocVecEnv(
            [
                lambda: Monitor(
                    AssistantWrapper(
                        gym.make(cfg.env.name),
                        assistant=make_pid_assistant(cfg),
                        # initial_mask=cfg.assistance.mask_schedule.get_mask(0),
                    ),
                    cfg.system.output_path,
                    allow_early_resets=True,
                )
                for i in range(cfg.train.num_envs)
            ]
        )
    else:
        # Only one env
        env = DummyVecEnv(
            [
                lambda: Monitor(
                    AssistantWrapper(
                        gym.make(cfg.env.name),
                        make_pid_assistant(cfg),
                        # initial_mask=cfg.assistance.mask_schedule.get_mask(0),
                    ),
                    cfg.system.output_path,
                    allow_early_resets=True,
                )
            ]
        )

    return env
