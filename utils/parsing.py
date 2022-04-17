import argparse
from datetime import datetime
import os
from dataclasses import dataclass

from utils.azuremlutils import get_azureml_mlflow_tracking_uri


@dataclass
class ExperimentConfig:
    """Class for the command line args"""
    exp_id: str  # Experiment ID
    num_envs: int
    timesteps: int
    output_path: str
    log_path: str
    mlflow_tracking_uri: str
    n_repetitions: int
    n_eval_episodes: int


def parse_experiment_info() -> ExperimentConfig:
    """Parser for the flags that can be passed with the run/train/test scripts."""
    # Get current datetime to use as default experiment ID if nothing else provided
    now = datetime.now()
    # Format string: yymmddHHMMss (year-w/o-century month day hour minute second)
    now_str = now.strftime(r"%y%m%d%H%M%s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id",
                        default=now_str,
                        type=str,
                        help="Which experiment number to run/train/test")
    parser.add_argument(
        "--num_envs",
        default=8,
        type=int,
        help="Number of parallell processes of environments to run")
    parser.add_argument("--timesteps",
                        default=int(5e6),
                        type=int,
                        help="Total number of timesteps to train")

    # The default values of output_path and log_path ("./logs", "./outputs") specifies folders where
    # Azure ML automatically saves the outputs and logs.
    # See https://docs.microsoft.com/en-us/azure/machine-learning/how-to-save-write-experiment-files#where-to-write-files
    parser.add_argument("--output_path",
                        default="./outputs",
                        type=str,
                        help="Folder to place trained agents")
    parser.add_argument("--log_path",
                        default="./logs",
                        type=str,
                        help="Folder to place tensorboard logs")
    parser.add_argument(
        "--mlflow_tracking_uri",
        default=None,
        type=str,
        help=
        "MLFlow tracking URI. For AzureML, it can be found using utils/azuremlutils.py"
    )
    parser.add_argument(
        "--n_repetitions",
        default=1,
        type=int,
        help="How many times to repeat the same experiment. Useful for running overnight etc."
    )
    parser.add_argument(
        "--n_eval_episodes",
        default=100,
        type=int,
        help="Number of episodes to run when evaluating the final agent."
    )
    

    args = parser.parse_args()

    experiment_config = ExperimentConfig(
        exp_id=args.exp_id,
        timesteps=args.timesteps,
        output_path=args.output_path,
        log_path=args.log_path,
        num_envs=args.num_envs,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        n_repetitions=args.n_repetitions,
        n_eval_episodes=args.n_eval_episodes)
        

    return experiment_config
