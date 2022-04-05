from dataclasses import dataclass

# Code borrowed from https://github.com/simentha/gym-auv/blob/master/gym_auv/utils/controllers.py
# and modified

class PID:
    """
    PID controller with support for anti-wind-up. 
    To deactivate anti-wind-up, set valid_input_range to None.
    """
    def __init__(self, Kp, Ki, Kd, timestep, valid_input_range=[-1.0, 1.0]):
        assert timestep > 0.0, f"timestep {timestep} is not a positive number!"
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.timestep = timestep
        if valid_input_range is not None:
            assert len(valid_input_range) == 2
            self.min_input, self.max_input = valid_input_range
        else:
            self.min_input, self.max_input = float("-inf"), float("inf")
        self.reset()
        
    def reset(self) -> None:
        self._u = 0
        self.accumulated_error = 0
        self.last_error = 0
        
    def u(self, error):
        if self._u < self.min_input or self.max_input < self._u: 
            # anti wind up, don't accumulate the error
            pass
        else: 
            self.accumulated_error += error * self.timestep

        derivative_error = (error-self.last_error) / self.timestep
        self._u = self.Kp * error + self.Ki * self.accumulated_error \
                + self.Kd * derivative_error
        return self._u
    
class PI(PID):
    def __init__(self, Kp, Ki, timestep):
        # Inherit from PID, set Kd to 0
        super(PI, self).__init__(Kp=Kp, Ki=Ki, Kd=0, timestep=timestep)
