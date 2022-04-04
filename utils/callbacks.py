from abc import ABC, abstractmethod
import os

from stable_baselines3.common.callbacks import BaseCallback, EventCallback


class AbstractTracker(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def register_model(self, path):
        pass


class SaveAndTrackCallback(BaseCallback):
    """
    A callback that logs models and performance periodically to a tracker
    A tracker for MLFlow is recommended, but the implementation supports using another library by
    implementing another tracker that inherits from AbstractTracker

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, save_dir, tracker=None, save_freq=100000, verbose=0):
        super(SaveAndTrackCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

        self.save_dir = save_dir
        self.save_freq = save_freq
        self.tracker = tracker

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        if self.n_calls % self.save_freq == 0:
            
            # Save the model to disk
            path = os.path.join(
                self.save_dir, "model_" + str(self.num_timesteps) + ".pkl")
            self.model.save(path)

            # Register model with tracker
            if self.tracker:
                self.tracker.log_artifact(path)

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # TODO: save the last model
        path = os.path.join(self.save_dir, "final_model.pkl")
        self.model.save(path)

        # TODO: Log some other metrics?

        # Register last model with tracker
        if self.tracker:
            self.tracker.log_artifact(path)


class SaveCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, agents_dir, verbose=0):
        super(SaveCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.agents_dir = agents_dir

        self.rollout_save_freq = 10  # Save agent every rollout_save_freq rollouts
        self.n_rollouts = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        # global n_steps, best_mean_reward
        # n_steps = self.num_timesteps

        # if (n_steps + 1) % 5 == 0:
        if self.n_rollouts % self.rollout_save_freq == 0:
            # Save agent every rollout_save_freq rollouts
            self.model.save(
                os.path.join(self.agents_dir,
                             "model_" + str(self.num_timesteps) + ".pkl"))
        self.rollout_save_freq += 1

        return True

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        pass

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass