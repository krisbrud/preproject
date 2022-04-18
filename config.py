# my_project/config.py
import os

from dataclasses import dataclass

from assisted_baselines.common.assistants.pid import PIDGains
from assisted_baselines.common.mask import BaseMaskSchedule
from assisted_baselines.common.schedules.checkpoint_schedule import CheckpointSchedule
from utils.auv_masks import mask_surge_only


@dataclass
class ExperimentConfig:
    name: str = "example_experiment_name"


@dataclass
class SystemConfig:
    # Log path. Required to be "./logs" by AzureML
    log_path: str = os.path.join(os.curdir, "logs")
    # Path for outputs. Required to be "./outputs" by AzureML for saving of artifacts
    output_path: str = os.path.join(os.curdir, "outputs")


@dataclass
class HyperparamConfig:
    # Also see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters
    # for more details
    n_steps: int = 1024
    learning_rate: float = 1e-3
    batch_size: int = 1024
    gae_lambda: float = 0.95
    gamma: float = 0.999  # Discount factor
    n_epochs = 4  # Number of epochs per rollout
    clip_range: float = 0.2  # Clip range for PPO objective function
    ent_coef: float = 0.01  # Coefficient for entropy loss
    verbose: int = 2


@dataclass
class TrainConfig:
    # Number of parallel environments to use with SubProcVecEnv
    num_envs = 8
    # Total timesteps to run training
    total_timesteps = int(30e6)  # int(100e3)
    # How many timesteps between each time agent is saved to disk and MLFlow
    save_freq = int(100e3)
    # MLFlow Tracking URI for logging metrics and artifacts.
    # Set to None if it's not going to be used.
    mlflow_tracking_uri = "azureml://northeurope.api.azureml.ms/mlflow/v1.0/subscriptions/3165a1c1-fd45-4c8d-938e-0058c823f960/resourceGroups/aml-playground/providers/Microsoft.MachineLearningServices/workspaces/aml-playground"
    # RL Algorithm to use. Currently supports "AssistedPPO", which is implmented in `krisbrud/assisted-baselines`.
    algorithm = "AssistedPPO"
    # How many evaluation episodes to run when evaluating the environment
    n_eval_episodes = 100


@dataclass
class AuvPidConfig:
    # Gains for PID controllers for assistant
    surge = PIDGains(Kp=2, Ki=1.5, Kd=0)
    rudder = PIDGains(Kp=3.5, Ki=0.05, Kd=0.03)
    elevator = PIDGains(Kp=3.5, Ki=0.05, Kd=0.03)


@dataclass
class AssistanceConfig:
    mask_schedule: BaseMaskSchedule
    # Define parameters for PID controllers
    auv_pid: AuvPidConfig = AuvPidConfig()


@dataclass
class EnvConfig:
    # Class for configuring the OpenAI gym env
    # Name of gym environment to look up
    name = "PathFollowAuv3D-v0"
    # Number of actuators in action space
    n_actions = 3


@dataclass
class Config:
    experiment: ExperimentConfig
    system: SystemConfig
    hyperparam: HyperparamConfig
    train: TrainConfig
    assistance: AssistanceConfig
    env: EnvConfig


def get_default_config() -> Config:
    # Since some hyperparameters and instantiations may depend on others, we do it
    # this way
    experiment = ExperimentConfig()
    system = SystemConfig()
    hyperparam = HyperparamConfig()
    train = TrainConfig()
    env = EnvConfig()

    # AssistanceConfig depends on the others, instantiate last
    assistance = AssistanceConfig(
        mask_schedule=CheckpointSchedule(
            {0: mask_surge_only}, total_timesteps=train.total_timesteps
        )
    )

    config = Config(
        experiment=experiment,
        system=system,
        hyperparam=hyperparam,
        train=train,
        assistance=assistance,
        env=env,
    )
    return config
