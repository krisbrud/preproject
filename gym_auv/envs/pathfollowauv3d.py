import numpy as np
import gym
import gym_auv.utils.geomutils as geom
import matplotlib.pyplot as plt
import skimage.measure

from gym_auv.objects.auv3d import AUV3D
from gym_auv.objects.current3d import Current
from gym_auv.objects.QPMI import QPMI, generate_random_waypoints
from gym_auv.objects.path3d import Path3D
from gym_auv.objects.obstacle3d import Obstacle
from gym_auv.utils.controllers import PI, PID


class PathFollowAuv3D(gym.Env):
    """
    3D Auv environment with path following (but without obstacles and sonar)
    """
    def __init__(self, env_config, control_structure="pid-assisted"):
        for key in env_config:
            setattr(self, key, env_config[key])

        self.n_observations = self.n_obs_states + self.n_obs_errors + self.n_obs_inputs
        self.action_space = gym.spaces.Box(
            low=np.array([-1] * self.n_actuators),
            high=np.array([1] * self.n_actuators),
            dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=np.array([-1] * self.n_observations),
            high=np.array([1] * self.n_observations),
            dtype=np.float32)

        valid_control_structures = ("end-to-end", "pid-assisted")
        assert control_structure in valid_control_structures, f"The control structure {control_structure}" \
            "is not supported, the supported control structures are: {valid_control_structures}"
        self.control_structure = control_structure

        self.reset()

    def reset(self):
        """
        Resets environment to initial state. 
        """
        self.vessel = None
        self.path = None
        self.u_error = None
        self.e = None
        self.h = None
        self.chi_error = None
        self.upsilon_error = None
        self.thrust_command = 0  # None
        self.elevator_command = 0  # None

        self.waypoint_index = 0
        self.prog = 0
        self.path_prog = []
        self.success = False

        self.obstacles = []
        self.nearby_obstacles = []
        self.sensor_readings = np.zeros(shape=self.sensor_suite,
                                        dtype=float)  # TODO Remove?
        self.collided = False
        self.penalize_control = 0.0

        self.observation = None
        self.action_derivative = np.zeros(self.n_actuators)
        self.past_states = []
        self.past_actions = []
        self.past_errors = []
        self.past_obs = []
        self.current_history = []
        self.time = []
        self.total_t_steps = 0
        self.reward = 0

        self.generate_environment()
        self.update_control_errors()
        self.observation = self.observe(np.zeros(6, dtype=float))
        return self.observation

    def generate_environment(self):
        """
        Generates environment with a vessel, potentially ocean current and a 3D path.
        """
        # Generate training/test scenario
        init_state = self.initialize_state()

        # Generate AUV
        self.vessel = AUV3D(self.step_size, init_state)
        self.thrust_controller = PI()
        self.elevator_controller = PID()

    def initialize_state(self):
        # Initialize the state of the auv and environment.
        # Similar to "beginner scenario" from "PathColav3D-v0"
        self.current = Current(mu=0,
                               Vmin=0,
                               Vmax=0,
                               Vc_init=0,
                               alpha_init=0,
                               beta_init=0,
                               t_step=0)  #Current object with zero velocity
        waypoints = generate_random_waypoints(self.n_waypoints)
        self.path = QPMI(waypoints)
        init_pos = [
            np.random.uniform(0, 2) * (-5),
            np.random.normal(0, 1) * 5,
            np.random.normal(0, 1) * 5
        ]
        init_attitude = np.array([
            0,
            self.path.get_direction_angles(0)[1],
            self.path.get_direction_angles(0)[0]
        ])
        initial_state = np.hstack([init_pos, init_attitude])
        return initial_state

    def plot_section3(self):
        plt.rc('lines', linewidth=3)
        ax = self.plot3D(wps_on=False)
        ax.set_xlabel(xlabel="North [m]", fontsize=14)
        ax.set_ylabel(ylabel="East [m]", fontsize=14)
        ax.set_zlabel(zlabel="Down [m]", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([-50, 0, 50])
        ax.set_zticks([-50, 0, 50])
        ax.view_init(elev=-165, azim=-35)
        ax.scatter3D(*self.vessel.position,
                     label="Initial Position",
                     color="y")

        self._axis_equal3d(ax)
        ax.legend(fontsize=14)
        plt.show()

    def step(self, action):
        """
        Simulates the environment one time-step. 
        """
        # Simulate Current
        self.current.sim()
        nu_c = self.current(self.vessel.state)
        self.current_history.append(nu_c[0:3])

        # Simulate AUV dynamics one time-step and save action and state
        self.update_control_errors()

        # if self.control_structure == "end-to-end":
        #     # Train all inputs simultaneously without assistance from PID controllers
        #     pass  # No changes to "action"-variable needed
        # elif self.control_structure == "pid-assisted":
        #     # Use PI controller for surge speed and PID controller
        #     # for elevator, while letting the RL agent control the rudder

        #     self.thrust_command = self.thrust_controller.u(self.u_error)
        #     self.rudder_command = action  # Action seen from agent is only rudder
        #     self.elevator_command = self.elevator_controller.u(
        #         self.upsilon_error)  # self.elevator_controller.u(self.)

        #     action = np.hstack((self.thrust_command, self.rudder_command,
        #                         self.elevator_command))
        # else:
        #     raise ValueError(
        #         f"Invalid value for control structure: \"{self.control_structure}\""
        #     )

        action = np.clip(action, np.array([0, -1, -1]), np.array([1, 1, 1]))
        if len(self.past_actions) > 0:
            self.action_derivative = (
                action[1:] - self.past_actions[-1][1:]) / (self.step_size)  # ?

        self.vessel.step(action, nu_c)

        self.past_states.append(np.copy(self.vessel.state))
        self.past_errors.append(
            np.array([
                self.u_error, self.chi_error, self.e, self.upsilon_error,
                self.h
            ]))
        self.past_actions.append(self.vessel.input)

        if self.path:
            self.prog = self.path.get_closest_u(self.vessel.position,
                                                self.waypoint_index)
            self.path_prog.append(self.prog)

            # Check if a waypoint is passed
            k = self.path.get_u_index(self.prog)
            if k > self.waypoint_index:
                # print("Passed waypoint {:d}".format(k + 1))
                self.waypoint_index = k

        # Calculate reward based on observation and actions
        done, step_reward = self.step_reward(self.observation, action)
        info = {}

        # Make next observation
        self.observation = self.observe(nu_c)
        self.past_obs.append(self.observation)

        # Save sim time info
        self.total_t_steps += 1
        self.time.append(self.total_t_steps * self.step_size)

        return self.observation, step_reward, done, info

    def observe(self, nu_c):
        """
        Returns observations of the environment. 
        """

        # TODO: use less magical indexing?
        obs = np.zeros((self.n_observations, ))

        # surge, sway, heave
        obs[0] = np.clip(self.vessel.relative_velocity[0] / 2, -1, 1)
        obs[1] = np.clip(self.vessel.relative_velocity[1] / 0.3, -1, 1)
        obs[2] = np.clip(self.vessel.relative_velocity[2] / 0.3, -1, 1)
        obs[3] = np.clip(self.vessel.roll / np.pi, -1, 1)
        obs[4] = np.clip(self.vessel.pitch / np.pi, -1, 1)
        obs[5] = np.clip(self.vessel.heading / np.pi, -1, 1)
        # roll rate, pitch rate, yaw rate
        obs[6] = np.clip(self.vessel.angular_velocity[0] / 1.2, -1, 1)
        obs[7] = np.clip(self.vessel.angular_velocity[1] / 0.4, -1, 1)
        obs[8] = np.clip(self.vessel.angular_velocity[2] / 0.4, -1, 1)
        obs[9] = np.clip(self.u_error / 2, -1, 1)  # surge error
        obs[10] = np.clip(self.chi_error / np.pi, -1, 1)  # course error
        obs[11] = np.clip(self.upsilon_error / np.pi, -1, 1)  # elevation error
        obs[12] = np.clip(self.e / 25, -1, 1)  # cross track error
        obs[13] = np.clip(self.h / 25, -1, 1)  # vertical track error
        obs[14] = self.thrust_command  # propeller shaft speed
        obs[15] = self.elevator_command  # stern plane position

        return obs

    def step_reward(self, obs, action):
        """
        Calculates the reward function for one time step. Also checks if the episode should end. 
        """
        done = False
        step_reward = 0

        reward_roll = self.vessel.roll**2 * self.reward_roll + self.vessel.angular_velocity[
            0]**2 * self.reward_rollrate
        reward_control = action[1]**2 * self.reward_use_rudder + action[
            2]**2 * self.reward_use_elevator
        reward_path_following = self.chi_error**2 * self.reward_heading_error + self.upsilon_error**2 * self.reward_pitch_error

        step_reward = self.lambda_reward * reward_path_following + reward_roll + reward_control
        self.reward += step_reward

        end_cond_1 = self.reward < self.min_reward
        end_cond_2 = self.total_t_steps >= self.max_t_steps
        end_cond_3 = np.linalg.norm(
            self.path.get_endpoint() - self.vessel.position
        ) < self.accept_rad and self.waypoint_index == self.n_waypoints - 2
        end_cond_4 = abs(self.prog - self.path.length) <= self.accept_rad / 2.0

        if end_cond_1 or end_cond_2 or end_cond_3 or end_cond_4:
            if end_cond_2:
                print("Max timesteps reached!")

            if end_cond_3:
                print("AUV reached target!")
                self.success = True

            if end_cond_4:
                print("Progression larger than end of path!")

            print("Episode finished after {} timesteps with reward: {}".format(
                self.total_t_steps, self.reward.round(1)))
            done = True
        return done, step_reward

    def update_control_errors(self):
        """
        Updates and sets the control errors.
        """
        # Update cruise speed error
        self.u_error = np.clip(
            (self.cruise_speed - self.vessel.relative_velocity[0]) / 2, -1, 1)
        self.chi_error = 0.0
        self.e = 0.0
        self.upsilon_error = 0.0  # TODO remove?
        self.h = 0.0

        # Get path course and elevation
        s = self.prog
        chi_p, upsilon_p = self.path.get_direction_angles(s)

        # Calculate tracking errors
        SF_rotation = geom.Rzyx(0, upsilon_p, chi_p)  # Serret-Frenet frame
        epsilon = np.transpose(SF_rotation).dot(self.vessel.position -
                                                self.path(self.prog))
        e = epsilon[1]
        h = epsilon[2]

        # Calculate course and elevation errors from tracking errors
        chi_r = np.arctan2(-e, self.la_dist)
        upsilon_r = np.arctan2(h, np.sqrt(e**2 + self.la_dist**2))
        chi_d = chi_p + chi_r
        upsilon_d = upsilon_p + upsilon_r
        self.chi_error = np.clip(
            geom.ssa(self.vessel.chi - chi_d) / np.pi, -1, 1)
        #self.e = np.clip(e/12, -1, 1)
        self.e = e
        self.upsilon_error = np.clip(
            geom.ssa(self.vessel.upsilon - upsilon_d) / np.pi, -1, 1)
        #self.h = np.clip(h/12, -1, 1)
        self.h = h

    def plot3D(self, wps_on=True):
        """
        Returns 3D plot of path and obstacles.
        """
        ax = self.path.plot_path(wps_on)
        return self._axis_equal3d(ax)

    def _axis_equal3d(self, ax):
        """
        Shifts axis in 3D plots to be equal. Especially useful when plotting obstacles, so they appear spherical.
        
        Parameters:
        ----------
        ax : matplotlib.axes
            The axes to be shifted. 
        """
        extents = np.array(
            [getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
        sz = extents[:, 1] - extents[:, 0]
        centers = np.mean(extents, axis=1)
        maxsize = max(abs(sz))
        r = maxsize / 2
        for ctr, dim in zip(centers, 'xyz'):
            getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        return ax
