import torch
from assisted_baselines.common.mask import ActiveActionsMask


AGENT_FLAG = True
ASSISTANT_FLAG = False
# Define masks
# Masks are named after the actuators that are controlled by the RL agent

# TODO: Find out whether to instantiate the masks or pass the same ones

# (surge, rudder, elevator)
mask_surge_only = ActiveActionsMask(
    3, torch.BoolTensor([AGENT_FLAG, ASSISTANT_FLAG, ASSISTANT_FLAG])
)
mask_surge_and_rudder = ActiveActionsMask(
    3, torch.BoolTensor([AGENT_FLAG, AGENT_FLAG, ASSISTANT_FLAG])
)
mask_agent_only = ActiveActionsMask(
    3, torch.BoolTensor([AGENT_FLAG, AGENT_FLAG, AGENT_FLAG])
)
mask_surge_and_elevator = ActiveActionsMask(
    3, torch.BoolTensor([AGENT_FLAG, ASSISTANT_FLAG, AGENT_FLAG])
)
mask_rudder_only = ActiveActionsMask(
    3, torch.BoolTensor([ASSISTANT_FLAG, AGENT_FLAG, ASSISTANT_FLAG])
)
mask_elevator_only = ActiveActionsMask(
    3, torch.BoolTensor([ASSISTANT_FLAG, ASSISTANT_FLAG, AGENT_FLAG])
)
mask_rudder_and_elevator = ActiveActionsMask(
    3, torch.BoolTensor([ASSISTANT_FLAG, AGENT_FLAG, AGENT_FLAG])
)
mask_assistant_only = ActiveActionsMask(
    3, torch.BoolTensor([ASSISTANT_FLAG, ASSISTANT_FLAG, ASSISTANT_FLAG])
)
