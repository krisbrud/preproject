# my_project/config.py
import os
from yacs.config import CfgNode as CN


_C = CN()

## EXPERIMENT CONFIG
_C.experiment = CN()
# Name of experiment for naming tensorboard logs and runs in MLFlow
_C.experiment.name = "example_experiment_name"

## SYSTEM CONFIG
_C.system = CN()
# Log path. Required to be "./logs" by AzureML
_C.system.log_path = os.path.join(os.curdir, "logs")
# Path for outputs. Required to be "./outputs" by AzureML for saving of artifacts
_C.system.output_path = os.path.join(os.curdir, "outputs")

## TRAINING CONFIG
_C.train = CN()
# Number of parallel environments to use with SubProcVecEnv
_C.train.num_envs = 8
# Total timesteps to run training
_C.train.total_timesteps = int(30e6)
# How many timesteps between each time agent is saved to disk and MLFlow
_C.train.save_freq = int(100e3)
# MLFlow Tracking URI for logging metrics and artifacts. 
# Set to None if it's not going to be used.
_C.train.mlflow_tracking_uri
# RL Algorithm to use. Currently supports "AssistedPPO", which is implmented in `krisbrud/assisted-baselines`.
_C.train.algorithm = "AssistedPPO"
# How many evaluation episodes to run when evaluating the environment
_C.train.n_eval_episodes

## ALGORITHM HYPERPARAMETERS
# Hyperparameters for PPO algorithm
_C.hyperparam = CN()
# Also see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html#parameters
# for more details
_C.hyperparam.n_steps = 1024
_C.hyperparam.learning_rate = 1e-3
_C.hyperparam.batch_size = 1024
_C.hyperparam.gae_lambda = 0.95
_C.hyperparam.gamma = 0.999 # Discount factor
_C.hyperparam.n_epochs = 4  # Number of epochs per rollout
_C.hyperparam.clip_range = 0.2  # Clip range for PPO objective function
_C.hyperparam.ent_coef = 0.01   # Coefficient for entropy loss
_C.hyperparam.verbose = 2

## ASSISTANCE CONFIG
# Configuration for assistants, masks and schedules
_C.assistance = CN()
_C.assistance.masks = CN()
AGENT_FLAG = True 
ASSISTANT_FLAG = False
# Define masks
# Masks are named after the actuators that are controlled by the RL agent
# (surge, rudder, elevator)
mask_surge_only =                [AGENT_FLAG,        ASSISTANT_FLAG,     ASSISTANT_FLAG]
mask_surge_and_rudder =          [AGENT_FLAG,        AGENT_FLAG,         ASSISTANT_FLAG]
mask_agent_only =                [AGENT_FLAG,        AGENT_FLAG,         AGENT_FLAG]
mask_surge_and_elevator =        [AGENT_FLAG,        ASSISTANT_FLAG,     AGENT_FLAG]
mask_rudder_only =               [ASSISTANT_FLAG,    AGENT_FLAG,         ASSISTANT_FLAG]
mask_elevator_only =             [ASSISTANT_FLAG,    ASSISTANT_FLAG,     AGENT_FLAG]
mask_rudder_and_elevator =       [ASSISTANT_FLAG,    AGENT_FLAG,         AGENT_FLAG]
mask_assistant_only =            [ASSISTANT_FLAG,    ASSISTANT_FLAG,     ASSISTANT_FLAG]

_C.assistance.checkpoints = {
    0: mask_elevator_only
}

## ENVIRONMENT CONFIG
_C.env = CN()
# Name of gym environment to look up
_C.env.name = "PathFollowAuv3D-v0"





def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

# Alternatively, provide a way to import the defaults as
# a global singleton:
# cfg = _C  # users can `from config import cfg`