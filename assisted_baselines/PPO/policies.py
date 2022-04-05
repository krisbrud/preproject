# This file is here just to define policies that work for AssistedPPO
from stable_baselines3.common.policies import (
    register_policy,
)

from assisted_baselines.common.assisted_actor_critic import AssistedActorCriticPolicy

AssistedPolicy = AssistedActorCriticPolicy

register_policy("AssistedPolicy", AssistedActorCriticPolicy)