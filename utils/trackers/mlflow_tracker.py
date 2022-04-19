from typing import Any, Dict
import mlflow
from pytablewriter import Bool

from utils.trackers.abstract_tracker import AbstractTracker


class MLFlowTracker(AbstractTracker):
    def __init__(self, mlflow_tracking_uri=None, experiment_name=None):
        super().__init__()

        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name

        # Initialize MLFlow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        mlflow.set_experiment(str(self.experiment_name))

        # End the active run if one exists from previous repetition:
        if mlflow.active_run() is not None:
            mlflow.end_run()

        mlflow.start_run()

    def log_artifact(self, path, infer_artifact_path=True) -> Bool:
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

    def log_metric(self, metric_name, value):
        try:
            mlflow.log_metric(metric_name, value)
        except Exception as e:
            # Failed to log metric
            print(
                f"Couldn't log params (metric_name, value) ({metric_name}, {value}) to MLFlow!"
            )
            print(f"Exception: {e}")
            return False

        return True

    def __del__(self):
        # Destructor. End the current run.
        mlflow.end_run()
