import warnings
from typing import Any, Dict, Optional, Tuple, Type, Union
import gym

import numpy as np
import torch as th
from gym import spaces
from gym.core import Env
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import (
    explained_variance,
    get_schedule_fn,
    obs_as_tensor,
)
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from assisted_baselines.common.assistant import AssistantWrapper, BaseAssistant
from assisted_baselines.common.assisted_actor_critic import AssistedActorCriticPolicy
from assisted_baselines.common.assisted_rollout_buffer import AssistedRolloutBuffer
from assisted_baselines.common.mask import BaseMaskSchedule
from utils.trackers.abstract_tracker import AbstractTracker
from utils.trackers.mlflow_tracker import MLFlowOutputFormat, MLFlowTracker


class AssistedPPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)
    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)
    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[AssistedActorCriticPolicy]],
        env: Union[GymEnv, str],
        action_mask_schedule: BaseMaskSchedule,
        # assistant: BaseAssistant, #TODO remove
        tracker: AbstractTracker = None,
        assistant_action_noise_std=0.1,
        assistant_available_probability: float = 0.2,
        learn_from_assistant_actions: bool = False,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        # Add the required kwargs to the AssistedPolicy automagically
        if policy_kwargs:
            assisted_policy_kwargs = {
                # "assistant": assistant,
                "action_mask_schedule": action_mask_schedule,
                **policy_kwargs,
            }
        else:
            assisted_policy_kwargs = {
                # "assistant": assistant,
                "action_mask_schedule": action_mask_schedule,
            }

        super(AssistedPPO, self).__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=False,  # use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=assisted_policy_kwargs,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            _init_setup_model=False,
            policy_base=AssistedActorCriticPolicy,
            supported_action_spaces=(
                spaces.Box,
                # spaces.Discrete,
                # spaces.MultiDiscrete,
                # spaces.MultiBinary,
            ),
        )

        self.tracker = tracker

        self._first_rollout = True

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert (
                buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        self.action_mask_schedule = action_mask_schedule
        # self.assistant = assistant

        self.assistant_exploit_mode = True
        self.assistant_available_probability = assistant_available_probability
        print("assistant_available_probability", self.assistant_available_probability)
        self.learn_from_assistant_actions = learn_from_assistant_actions
        self.assistant_action_noise_std = assistant_action_noise_std
        # Wrap the environment with the Assistantwrapper
        # self.env = AssistantWrapper(self.env, assistant)

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(AssistedPPO, self)._setup_model()

        # Overwrite rollout buffer, as we need the AssistedRolloutBuffer, not
        # the normal RolloutBuffer from sb3
        self.rollout_buffer = AssistedRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, "
                    "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        return_vals = super()._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        # Add MLFlow-tracker output format to logger, such that metrics we log
        if self.tracker is not None and isinstance(self.tracker, MLFlowTracker):
            # Make the logger log metrics from the training monitor to MLFlow as well
            self.logger.output_formats.append(
                MLFlowOutputFormat(mlflow_tracker=self.tracker)
            )

        return return_vals

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        print(
            "Entered assisted ppo train. num_timesteps:",
            self.num_timesteps,
            "total timesteps",
            self._total_timesteps,
        )
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        first_batch = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                is_agent_chosen = rollout_data.is_agent_chosen[:, 0]

                (
                    values,
                    log_prob,
                    entropy,
                    assistant_values,
                ) = self.policy.evaluate_actions(rollout_data.observations, actions)

                values = values.flatten()
                assistant_values = assistant_values.flatten()
                is_agent_chosen = rollout_data.is_agent_chosen.flatten()

                # Normalize advantage
                # learn_from_assistant_actions = False  # True  # False

                if self.learn_from_assistant_actions:
                    advantages = rollout_data.advantages
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)
                else:
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)[
                        is_agent_chosen
                    ]
                    # ratio between old and new policy, should be one at the first iteration
                    advantages = rollout_data.advantages[is_agent_chosen]

                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                if self.learn_from_assistant_actions:
                    # Rescale the advantages corresponding to assistant actions with the "probability"
                    # i.e. pdf evaluated at pi_old(a_t | s_t) to deal with
                    advantage_weights = th.where(
                        is_agent_chosen,
                        th.ones_like(advantages),
                        th.exp(rollout_data.old_log_prob),
                    )
                    # Advantages are kept the same where agent action was taken.
                    advantages = advantages * advantage_weights

                if self._first_rollout and first_batch and False:
                    first_batch = False
                    print("batch size", self.batch_size)

                    print("\nShapes:")
                    print("actions\t\t", actions.shape)
                    print("observations\t", rollout_data.observations.shape)
                    print("values\t\t", values.shape)
                    print("assistant_values\t", assistant_values.shape)
                    print("advantages\t", advantages.shape)
                    print("log_prob\t", log_prob.shape)
                    print("ratio\t\t", ratio.shape)
                    print("is_agent_chosen", rollout_data.is_agent_chosen.shape)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                # policy_loss_1.retain_grad()
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                # policy_loss.retain_grad()
                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    # Make sure we optimize over the correct value heads by filtering by where which action was taken
                    mixed_values = th.where(is_agent_chosen, values, assistant_values)
                    # print("mixed values size", mixed_values.size())
                    values_pred = mixed_values  # mixed_values
                    # values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    raise NotImplementedError()  # TODO remove code block
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                # value_loss.retain_grad()
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = (
                        th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()

                # print("loss", loss)
                # print("loss grad", loss.grad)

                # print("policy loss 1", policy_loss_1)
                # print("policy loss 1 grad max", th.max(policy_loss_1.grad))
                # print("policy loss 1 grad min", th.min(policy_loss_1.grad))
                # print("policy loss 1 size", policy_loss_1.size())
                # print("policy loss 1 grad", policy_loss_1.grad)

                # Clip grad norm
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )

        # Train is called once per rollout - get and set assistant mask for next rollout
        self._first_rollout = False
        # self.policy.update_mask(timestep=self.num_timesteps)

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: AssistedRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy, assistant and mask and
        fill a ``RolloutBuffer``.

        Code is taken from OnPolicyAlgorithm and modified to support masking
        and assistants. https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py

        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        self.update_mask()

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        # if self.use_sde:
        #     self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()
        assistance_rates = []


        # print(f"type {type(self.env)}")
        if isinstance(self.env, VecEnv):
            assert env.env_is_wrapped(
                AssistantWrapper
            ), "VecEnv environments are not wrapped by a AssistantWrapper!"
        else:
            assert isinstance(
                self.env, AssistantWrapper
            ), "Environment isn't wrapped with AssistantWrapper!"

        # assert isinstance(self.env, AssistantWrapper)
        assert isinstance(self.env, VecEnv)
        assert isinstance(self.policy, AssistedActorCriticPolicy)

        rollout_agent_actions = []
        rollout_assistant_actions = []

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                agent_actions, agent_values, log_probs, assistant_values = self.policy(
                    obs_tensor
                )

            # if self.assistant_exploit_mode:
            # with th.no_grad():
                # Compare agent and assistant values
                # We slice the tensor to avoid an unnecessary dimension at the end. TODO
                is_agent_better_pred = th.gt(agent_values, assistant_values).cpu()

                # Draw from a bernoulli distribution whether we are allowed to use the assistant
                # this timestep
                is_assistant_available = th.bernoulli(
                    th.full_like(
                        is_agent_better_pred,
                        fill_value=self.assistant_available_probability,
                        dtype=th.float32,
                    )
                ).bool()
                # print("is assistant avail", is_assistant_available)

                # Choose agent in env if
                # it is predicted to be better OR the assistant is NOT available
                is_agent_chosen = is_agent_better_pred | (~is_assistant_available)

                # Calculate assistance rate
                assistance_rate = 1.0 - is_agent_chosen.float().mean().item()

                assistant_actions = self.env.env_method("get_assistant_action")
                # print("agent_actions", agent_actions)
                # print("assistant_actions", assistant_actions)
                assistant_actions = th.from_numpy(
                    np.stack(assistant_actions, axis=0).astype(np.float32)
                )

                assistant_actions_noise = th.normal(
                    mean=th.zeros_like(assistant_actions),
                    std=(
                        self.assistant_action_noise_std
                        * th.ones_like(assistant_actions)
                    ),
                )

                rollout_agent_actions.append(agent_actions.numpy())
                rollout_assistant_actions.append(assistant_actions.numpy())

                # rollout_assistant_actions.append(noisy_assistant_actions.numpy())

                actions = (
                    th.where(is_agent_chosen, agent_actions, assistant_actions)
                    .cpu()
                    .numpy()
                )

            assistance_rates.append(assistance_rate)

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(
                    actions, self.action_space.low, self.action_space.high
                )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            # TODO: Only if in exploit mode
            values = th.where(is_agent_chosen, agent_values, assistant_values)

            # print("type rolloutbuffer", type(rollout_buffer))

            # print("actions shape", actions.shape)
            assert isinstance(actions, np.ndarray), "actions not ndarray!"

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,  # TODO
                log_probs,
                is_agent_chosen[:, 0],
            )
            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        assistance_rate_mean = np.mean(assistance_rates)
        self.logger.record("rollout/assistance_rate", assistance_rate_mean)

        print(
            "rollout agent action mean",
            np.mean(np.array(rollout_agent_actions), axis=0),
        )
        print(
            "rollout assistant action mean",
            np.mean(np.array(rollout_assistant_actions), axis=0),
        )

        # TODO: Toggle between exploration and exploitation scheme at random with rate p
        # For now: Toggle between modes explore and exploit

        self.assistant_exploit_mode = not self.assistant_exploit_mode

        callback.on_rollout_end()

        return True

    def update_mask(self):
        mask = self.action_mask_schedule.get_mask(timestep=self.num_timesteps)

        self.policy.set_mask(mask)
        # self.env.env_method("set_mask", mask)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "AssistedPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "AssistedPPO":

        return super(AssistedPPO, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )
