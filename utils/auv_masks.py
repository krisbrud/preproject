AGENT_FLAG = True
ASSISTANT_FLAG = False
# Define masks
# Masks are named after the actuators that are controlled by the RL agent

# TODO: Find out whether to instantiate the masks or pass the same ones

# (surge, rudder, elevator)
mask_surge_only = [AGENT_FLAG, ASSISTANT_FLAG, ASSISTANT_FLAG]
mask_surge_and_rudder = [AGENT_FLAG, AGENT_FLAG, ASSISTANT_FLAG]
mask_agent_only = [AGENT_FLAG, AGENT_FLAG, AGENT_FLAG]
mask_surge_and_elevator = [AGENT_FLAG, ASSISTANT_FLAG, AGENT_FLAG]
mask_rudder_only = [ASSISTANT_FLAG, AGENT_FLAG, ASSISTANT_FLAG]
mask_elevator_only = [ASSISTANT_FLAG, ASSISTANT_FLAG, AGENT_FLAG]
mask_rudder_and_elevator = [ASSISTANT_FLAG, AGENT_FLAG, AGENT_FLAG]
mask_assistant_only = [ASSISTANT_FLAG, ASSISTANT_FLAG, ASSISTANT_FLAG]
