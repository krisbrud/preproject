import gym
import numpy as np
import glob

from pandas import DataFrame
from stable_baselines3 import PPO
from gym_auv.utils.controllers import PI, PID
from utils.plotting import plot_3d, plot_angular_velocity, plot_attitude, plot_control_errors, plot_control_inputs, plot_current_data, plot_velocity

PI = PI()
PID_cross = PID(Kp=1.8, Ki=0.01, Kd=0.035)

def find_tb_logs(dir_to_search):
    tb_log_paths = glob.glob(f"{dir_to_search}/tensorboard/**/events.out.tfevents*")
    return tb_log_paths



def calculate_IAE(sim_df):
    """
    Calculates and prints the integral absolute error provided an environment id and simulation data
    """
    IAE_cross = sim_df[r"e"].abs().sum()
    IAE_vertical = sim_df[r"h"].abs().sum()
    print("IAE Cross track: {}, IAE Vertical track: {}".format(
        IAE_cross, IAE_vertical))
    return IAE_cross, IAE_vertical


def simulate_environment(env, agent):
    global error_labels, current_labels, input_labels, state_labels
    state_labels = [
        r"$N$", r"$E$", r"$D$", r"$\phi$", r"$\theta$", r"$\psi$", r"$u$",
        r"$v$", r"$w$", r"$p$", r"$q$", r"$r$"
    ]
    current_labels = [r"$u_c$", r"$v_c$", r"$w_c$"]
    input_labels = [r"$\eta$", r"$\delta_r$", r"$\delta_s$"]
    error_labels = [
        r"$\tilde{u}$", r"$\tilde{\chi}$", r"e", r"$\tilde{\upsilon}$", r"h"
    ]
    labels = np.hstack(
        ["Time", state_labels, input_labels, error_labels, current_labels])

    done = False
    env.reset()
    while not done:
        action = agent.predict(env.observation, deterministic=True)[0]
        _, _, done, _ = env.step(action)
    errors = np.array(env.past_errors)
    time = np.array(env.time).reshape((env.total_t_steps, 1))
    sim_data = np.hstack(
        [time, env.past_states, env.past_actions, errors, env.current_history])
    df = DataFrame(sim_data, columns=labels)
    error_labels = [r"e", r"h"]
    return df


def simulate_and_plot_agent(agent, env_name="PathFollowAuv3D-v0"):
    env = gym.make(env_name)
    sim_df = simulate_environment(env, agent)
    sim_df.to_csv(r'simdata.csv')
    calculate_IAE(sim_df)
    plot_attitude(sim_df)
    plot_velocity(sim_df)
    plot_angular_velocity(sim_df)
    plot_control_inputs([sim_df])
    plot_control_errors([sim_df])
    plot_3d(env, sim_df)
    plot_current_data(sim_df)