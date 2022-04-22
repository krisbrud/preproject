import dataclasses

from assisted_baselines.common.schedules.checkpoint_schedule import CheckpointSchedule
from assisted_baselines.PPO import AssistedPPO

from config import get_config

from utils.auv_masks import mask_assistant_only
from utils.assisted_env_factory import get_eval_env
import utils.utils


def run_and_plot_assistant_only():
    cfg = get_config()
    # Initialize environment and assistant agent
    cfg.assistance.mask_schedule = CheckpointSchedule(
        {0: mask_assistant_only}, total_timesteps=cfg.train.total_timesteps
    )
    cfg.train.num_envs = 1  # Only one env needed
    cfg.assistance.mask_schedule = CheckpointSchedule(
        {0: mask_assistant_only}, total_timesteps=cfg.train.total_timesteps
    )
    # env = get_assisted_envs(cfg)
    env = get_eval_env(cfg)
    hyperparams = dataclasses.asdict(cfg.hyperparam)

    agent = AssistedPPO(
        "AssistedPolicy", env, cfg.assistance.mask_schedule, **hyperparams
    )

    plot_dir = cfg.system.output_path

    utils.utils.simulate_and_plot_agent(agent=agent, env=env, plot_dir=plot_dir)
