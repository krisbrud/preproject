# from bisect import bisect, bisect_left
from typing import Dict, List

from assisted_baselines.common.mask import ActiveActionsMask, BaseMaskSchedule


class SequentialSchedule(BaseMaskSchedule):
    """
    SequentialSchedule sequentially goes through the masks each time get_mask is called.

    :param total_timesteps - the total number of timesteps to train for
    """
    def __init__(self, masks: List[ActiveActionsMask], total_timesteps) -> None:
        super(SequentialSchedule, self).__init__()

        self.total_timesteps = total_timesteps
        self.masks = masks

        self.current_idx = 0

        assert len(self.masks) > 0

        for mask in self.masks:
            assert isinstance(mask, ActiveActionsMask)
        # for timestep, mask in self.checkpoints.items():
        #     assert timestep >= 0
        #     assert isinstance(timestep, int)
        #     assert isinstance(mask, ActiveActionsMask)

    def get_mask(self, timestep: int) -> ActiveActionsMask:
        assert timestep >= 0, f"Timestep {timestep} is negative!"

        self.current_idx = (self.current_idx + 1) % len(self.masks)

        mask = self.masks[self.current_idx]
        print(f"Mask: {mask.mask}")
        return mask

