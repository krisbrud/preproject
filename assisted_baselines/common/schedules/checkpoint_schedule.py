from bisect import bisect, bisect_left
from typing import Dict

from assisted_baselines.common.mask import ActiveActionsMask, BaseMaskSchedule


class CheckpointSchedule(BaseMaskSchedule):
    """
    CheckpointSchedule takes in a dictionary of timesteps as a key and the corresponding ActiveActionsMask as value. 
    The first policy must have the key 0.

    :param checkpoints - a dictionary of integers indicating the ActiveActionMask to use at the start of a rollout after the given timestep
    :param total_timesteps - the total number of timesteps to train for
    """
    def __init__(self, checkpoints: Dict[int, ActiveActionsMask], total_timesteps: int) -> None:
        super(CheckpointSchedule, self).__init__()

        self.checkpoints = checkpoints
        self.total_timesteps = total_timesteps

        for timestep, mask in self.checkpoints.items():
            assert timestep >= 0
            assert isinstance(timestep, int)
            assert isinstance(mask, ActiveActionsMask)

    def get_mask(self, timestep: int) -> ActiveActionsMask:
        assert timestep >= 0, f"Timestep {timestep} is negative!"

        try:
            mask = self.checkpoints[timestep]
            return mask
        except KeyError:
            pass
        
        sorted_keys_descending = sorted(list(self.checkpoints.keys()), reverse=True)
        
        for cp_timestep in sorted_keys_descending:
            if timestep >= cp_timestep:
                mask = self.checkpoints[cp_timestep]
                return mask
        
        raise ValueError(f"Couldn't find a mask for timestep {timestep}!")


        
