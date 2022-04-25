# Code borrowed from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py
# and modified

import collections
import copy
import warnings

# from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import gym.spaces
import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.distributions import (
    BernoulliDistribution,
    CategoricalDistribution,
    DiagGaussianDistribution,
    Distribution,
    MultiCategoricalDistribution,
    StateDependentNoiseDistribution,
    make_proba_distribution,
    sum_independent_dims,
)

from stable_baselines3.common.policies import BasePolicy

# from stable_baselines3.common.preprocessing import get_action_dim, is_image_space, maybe_transpose, preprocess_obs
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    # CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.utils import (
    get_device,
    is_vectorized_observation,
    obs_as_tensor,
)
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer

from assisted_baselines.common.assistant import AssistantWrapper, BaseAssistant
from assisted_baselines.common.mask import ActiveActionsMask, BaseMaskSchedule


class AssistedActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.
    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param assistant: TODO
    :param action_mask_schedule: TODO
    :param initial_timestep: The timestep when initializing the AssistedActorCriticPolicy.
        Used to get the correct mask when saving the policy to non-volatile storage and
        reopening it again.
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param ortho_init: Whether to use or not orthogonal initialization
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param squash_output: Whether to squash the output using a tanh function,
        this allows to ensure boundaries when using gSDE.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        # assistant: BaseAssistant,
        action_mask_schedule: BaseMaskSchedule,
        initial_timestep: int = 0,
        net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        # sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        # squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if use_sde:
            raise NotImplementedError(
                "SDE is not implemented for AssistedActorCriticPolicy!"
            )

        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(AssistedActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            # squash_output=squash_output,
        )

        # Default network architecture, from stable-baselines
        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [dict(pi=[64, 64], vf=[64, 64])]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.ortho_init = ortho_init

        self.features_extractor = features_extractor_class(
            self.observation_space, **self.features_extractor_kwargs
        )
        self.features_dim = self.features_extractor.features_dim

        self.normalize_images = normalize_images
        self.log_std_init = log_std_init
        dist_kwargs = None
        # Keyword arguments for gSDE distribution
        # if use_sde:
        # raise NotImplementedError
        # dist_kwargs = {
        #     "full_std": full_std,
        #     "squash_output": squash_output,
        #     "use_expln": use_expln,
        #     "learn_features": False,
        # }

        # if sde_net_arch is not None:
        # warnings.warn("sde_net_arch is deprecated and will be removed in SB3 v2.4.0.", DeprecationWarning)

        # self.use_sde = use_sde
        self.dist_kwargs = dist_kwargs

        # self.assistant: BaseAssistant = assistant
        self.action_mask_schedule: BaseMaskSchedule = action_mask_schedule

        # self.update_mask(timestep=initial_timestep)

        # assert self.assistant.action_space.shape == action_space.shape, "Assistant has different action space shape than policy!"

        # Action distribution
        self.action_dist = make_proba_distribution(
            action_space,
            # use_sde=use_sde,
            dist_kwargs=dist_kwargs,
        )

        self._build(lr_schedule)

    def set_mask(self, mask):
        print("updating policy mask")
        self.current_mask = mask

    def set_training_mode(self, mode: bool) -> None:
        # Since we want to have a mask in the assistant wrapper when evaluating,
        # but not when training (because of performance)
        # this is a neat way to achieve this without modifying other code too much.
        return super().set_training_mode(mode)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        default_none_kwargs = self.dist_kwargs or collections.defaultdict(lambda: None)

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.activation_fn,
                # use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                squash_output=default_none_kwargs["squash_output"],
                # full_std=default_none_kwargs["full_std"],
                use_expln=default_none_kwargs["use_expln"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, n_envs: int = 1) -> None:
        """
        Sample new weights for the exploration matrix.
        :param n_envs:
        """
        assert isinstance(
            self.action_dist, StateDependentNoiseDistribution
        ), "reset_noise() is only available when using gSDE"
        self.action_dist.sample_weights(self.log_std, batch_size=n_envs)

    def _build_mlp_extractor(self) -> None:
        """
        Create the policy and value networks.
        Part of the layers can be shared.
        """
        # Note: If net_arch is None and some features extractor is used,
        #       net_arch here is an empty list and mlp_extractor does not
        #       really contain any layers (acts like an identity module).
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.
        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, DiagGaussianDistribution):
            self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init
            )
            self.action_net_value_head = nn.Linear(
                latent_dim_pi, self.action_space.shape[0]
            )
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     self.action_net, self.log_std = self.action_dist.proba_distribution_net(
        #         latent_dim=latent_dim_pi, latent_sde_dim=latent_dim_pi, log_std_init=self.log_std_init
        #     )
        # elif isinstance(self.action_dist, (CategoricalDistribution, MultiCategoricalDistribution, BernoulliDistribution)):
        #     self.action_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)

        self.assistant_value_net = nn.Linear(
            self.mlp_extractor.latent_dim_vf, 1
        )  # Identical to value net

        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            # TODO: check for features_extractor
            # Values from stable-baselines.
            # features_extractor/mlp values are
            # originally from openai/baselines (default gains/init_scales).
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.action_net: 0.01,
                self.value_net: 1,
                self.assistant_value_net: 1,
            }
            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def forward(
        self, obs: th.Tensor, deterministic: bool = False
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Forward pass in all the networks (actor and critic)
        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Get the assistants actions
        # assistant_actions

        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        # Evaluate the values for the given observations
        agent_values = self.value_net(latent_vf)
        assistant_values = self.assistant_value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, agent_values, log_prob, assistant_values

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.
        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """
        mean_actions = self.action_net(latent_pi)

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        # elif isinstance(self.action_dist, CategoricalDistribution):
        #     # Here mean_actions are the logits before the softmax
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, MultiCategoricalDistribution):
        #     # Here mean_actions are the flattened logits
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, BernoulliDistribution):
        #     # Here mean_actions are the logits (before rounding to get the binary actions)
        #     return self.action_dist.proba_distribution(action_logits=mean_actions)
        # elif isinstance(self.action_dist, StateDependentNoiseDistribution):
        #     return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.
        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """
        actions = self.get_distribution(observation).get_actions(
            deterministic=deterministic
        )

        # Since the AssistantWrapper takes care of the assistant actions, we only pass
        # the actions taken from the distribution

        return actions

    def evaluate_actions(
        self, obs: th.Tensor, actions: th.Tensor, mask=None
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def evaluate_masked_actions(
        self, obs: th.Tensor, actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Evaluate actions according to the current policy,
        given the observations.
        :param obs:
        :param actions:
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)

        # Mask out the relevant actions for the epoch and calculate according to that
        masked_gaussian = self._get_masked_distribution(distribution)
        masked_actions = self.current_mask.select(actions)
        log_prob = sum_independent_dims(masked_gaussian.log_prob(masked_actions))
        entropy = sum_independent_dims(masked_gaussian.entropy())

        values = self.value_net(latent_vf)
        return values, log_prob, entropy

    # def action_net_value_estimate(self, obs: th.Tensor) -> th.Tensor:
    #     """
    #     Get the value estimate from the action net's auxillary value head. Useful for PPG.

    #     :param obs:
    #     :return: action net value head estimate
    #     """
    #     features = self.extract_features(obs)
    #     latent_pi, _ = self.mlp_extractor(features)
    #     pi_value_estimate = self.action_net_value_head(latent_pi)
    #     return pi_value_estimate

    def _get_masked_distribution(
        self, distribution: DiagGaussianDistribution
    ) -> th.distributions.Normal:
        masked_loc = self.current_mask.select(distribution.distribution.loc)
        masked_scale = self.current_mask.select(distribution.distribution.scale)

        masked_dist = th.distributions.Normal(loc=masked_loc, scale=masked_scale)
        return masked_dist

    def get_distribution(self, obs: th.Tensor) -> Distribution:
        """
        Get the current policy distribution given the observations.
        :param obs:
        :return: the action distribution.
        """
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._get_action_dist_from_latent(latent_pi)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.
        :param obs:
        :return: the estimated values.
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def predict_agent_and_expert_values(
        self, obs: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Get the estimated values of both the agent's policy network as well as the
        predicted value of the return of taking a step from the assistant given the current
        observation
        :param obs:
        :return: estimated values of agent actions, estimated value of expert actions
        """
        features = self.extract_features(obs)
        latent_vf = self.mlp_extractor.forward_critic(features)

        agent_values = self.value_net(latent_vf)
        assistant_values = self.assistant_value_net(latent_vf)

        return agent_values, assistant_values
