from typing import Any, Dict, Optional, Tuple, Union
import gym
import mlflow
import numpy as np
from pytablewriter import Bool
from stable_baselines3.common.logger import KVWriter

from utils.trackers.abstract_tracker import AbstractTracker


class MLFlowTracker(AbstractTracker):
    def __init__(self, mlflow_tracking_uri=None, experiment_name=None):
        super().__init__()

        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name

        # Initialize MLFlow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=str(self.experiment_name))

        # End the active run if one exists from previous repetition:
        if mlflow.active_run() is not None:
            mlflow.end_run()

        mlflow.start_run()

    def log_artifact(self, path: str, infer_artifact_path: Bool = True) -> Bool:
        """
        :param path (str) Path of artifact to plot
        :param infer_artifact_path (Bool) Automatically put .pkl files in "model" folder, plots in "plots" folder
        in mlflow artifacts
        """

        artifact_path = ""

        if infer_artifact_path:
            if path.endswith(".pkl"):
                artifact_path = "models"
            elif path.endswith(".png"):
                artifact_path = "plots"

        try:
            if artifact_path:
                mlflow.log_artifact(path, artifact_path=artifact_path)
            else:
                mlflow.log_artifact(path)
        except Exception as e:
            # Failed to log artifact
            print(f"Couldn't log artifact located at {path} to mlflow!")
            print(f"Exception: {e}")
            return False

        return True

    def log_param(self, key, value):
        try:
            mlflow.log_param(key, value)
        except Exception as e:
            # Failed to log param
            print(f"Couldn't log param (key, value) ({key}, {value}) to MLFlow!")
            print(f"Exception: {e}")
            return False

        return True

    def log_params(
        self, params: Dict[str, Any], prefix: str = "", prefix_delimiter: str = "."
    ):
        if prefix:  # No need if no prefix
            # Add prefix to string, such that similar keys/params may be grouped together
            # when looking through experiment in mlflow
            prefixed_params = dict()
            for k, v in params.items():
                prefixed_params[prefix + prefix_delimiter + k] = v

            params = prefixed_params

        try:
            mlflow.log_params(params)
        except Exception as e:
            # Failed to log params
            print(f"Couldn't log params ({params}) to MLFlow!")
            print(f"Exception: {e}")
            return False

        return True

    def log_metric(self, metric_name, value, step=None):
        try:
            mlflow.log_metric(metric_name, value, step=None)
        except Exception as e:
            # Failed to log metric
            print(
                f"Couldn't log metric (metric_name, value) ({metric_name}, {value}) to MLFlow!"
            )
            print(f"Exception: {e}")
            return False

        return True

    def log_metrics(self, metrics, step):
        try:
            mlflow.log_metrics(metrics, step)
        except Exception as e:
            # Failed to log metric. Don't crash the program.
            print(
                f"Couldn't log metrics (metric_name, value) ({metrics}) at timestep {step} to MLFlow!"
            )
            print(f"Exception: {e}")
            return False

        return True

    def __del__(self):
        # Destructor. End the current run.
        mlflow.end_run()


class MLFlowOutputFormat(KVWriter):
    def __init__(self, mlflow_tracker: MLFlowTracker) -> None:
        super().__init__()
        self.mlflow_tracker = mlflow_tracker

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        """
        Write a dictionary to MLFlow

        :param key_values:
        :param key_excluded:
        :param step:
        """
        metrics_to_log = dict()

        # Only log numerical metrics
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            # Based on the implementation for the TensorboardOutputFormat, as the use case
            # is similar, e.g. we do not want to log total_timesteps every rollout etc
            if excluded is not None and "tensorboard" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if isinstance(value, str):
                    # str is considered a np.ScalarType
                    pass  # Logging strings as metric is not allowed
                else:
                    metrics_to_log[key] = value

        # Log all metrics to MLFlow
        self.mlflow_tracker.log_metrics(metrics_to_log, step)

    def close(self):
        # Do nothing
        pass
