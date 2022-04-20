from abc import ABC, abstractmethod


class AbstractTracker(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def log_artifact(self, path):
        pass

    @abstractmethod
    def log_param(self, key, value):
        pass

    @abstractmethod
    def log_params(self, params):
        pass
    
    @abstractmethod
    def log_metric(self, metric_name, value):
        pass