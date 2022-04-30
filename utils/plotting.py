import os
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np


def set_default_plot_rc():
    """Sets the style for the plots report-ready"""
    colors = cycler(
        color=["#EE6666", "#3388BB", "#88DD89", "#EECC55", "#88BB44", "#FFBBBB"]
    ) + cycler(linestyle=["-", "-", "-", "--", ":", "-."])
    plt.rc(
        "axes",
        facecolor="#ffffff",
        edgecolor="black",
        axisbelow=True,
        grid=True,
        prop_cycle=colors,
    )
    plt.rc("grid", color="gray", linestyle="--")
    plt.rc("xtick", direction="out", color="black", labelsize=14)
    plt.rc("ytick", direction="out", color="black", labelsize=14)
    plt.rc("patch", edgecolor="#ffffff")
    plt.rc("lines", linewidth=4)


def plot_attitude(sim_df, output_dir, tracker=None):
    """Plots the state trajectories for the simulation data"""
    set_default_plot_rc()
    plt.figure()
    ax = sim_df.plot(x="Time", y=[r"$\phi$", r"$\theta$", r"$\psi$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Angular position [rad]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-np.pi, np.pi])
    # plt.show()
    path = os.path.join(output_dir, "attitude.png")
    plt.savefig(path)
    if tracker:
        tracker.log_artifact(path)


def plot_velocity(sim_df, output_dir, tracker=None):
    """Plots the velocity trajectories for the simulation data"""
    set_default_plot_rc()
    plt.figure()
    ax = sim_df.plot(x="Time", y=[r"$u$", r"$v$"], kind="line")
    ax.plot(
        sim_df["Time"], sim_df[r"$w$"], dashes=[3, 3], color="#88DD89", label=r"$w$"
    )
    ax.plot([0, sim_df["Time"].iloc[-1]], [1.5, 1.5], label=r"$u_d$")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Velocity [m/s]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-0.25, 2.25])
    # plt.show()
    path = os.path.join(output_dir, "velocity.png")
    plt.savefig(path)
    if tracker:
        tracker.log_artifact(path)


def plot_angular_velocity(sim_df, output_dir, tracker=None):
    """Plots the angular velocity trajectories for the simulation data"""
    set_default_plot_rc()
    plt.figure()
    ax = sim_df.plot(x="Time", y=[r"$p$", r"$q$", r"$r$"], kind="line")
    ax.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax.set_ylabel(ylabel="Angular Velocity [rad/s]", fontsize=14)
    ax.legend(loc="lower right", fontsize=14)
    ax.set_ylim([-1, 1])
    # plt.show()
    path = os.path.join(output_dir, "angular_velocity.png")
    plt.savefig(path)
    if tracker:
        tracker.log_artifact(path)


def plot_control_inputs(sim_dfs, output_dir, tracker=None):
    """Plot control inputs from simulation data"""
    set_default_plot_rc()
    plt.figure()
    c = ["#EE6666", "#88BB44", "#EECC55"]
    for i, sim_df in enumerate(sim_dfs):
        # control = np.sqrt(sim_df[r"$\delta_r$"]**2 + sim_df[r"$\delta_s$"]**2)
        # plt.plot(sim_df["Time"], sim_df[r"$\delta_s$"], linewidth=4, color=c[i])
        plt.plot(
            sim_df["Time"],
            sim_df[[r"$\eta$", r"$\delta_r$", r"$\delta_s$"]],
            linewidth=4,
            # color=c[i],
        )

    plt.xlabel(xlabel="Time [s]", fontsize=14)
    plt.ylabel(ylabel="Normalized Input", fontsize=14)
    plt.legend(loc="lower right", fontsize=14)
    plt.legend(
        # [r"$\lambda_r=0.9$", r"$\lambda_r=0.5$", r"$\lambda_r=0.1$"],
        [r"$\eta$", r"$\delta_r$", r"$\delta_s$"],
        loc="upper right",
        fontsize=14,
    )
    plt.ylim([-1.25, 1.25])
    path = os.path.join(output_dir, "control_inputs.png")
    plt.savefig(path)
    if tracker:
        tracker.log_artifact(path)
    # plt.show()


def plot_control_errors(sim_dfs, output_dir, tracker=None):
    """
    Plot control inputs from simulation data
    """
    # error_labels = [r'e', r'h']
    set_default_plot_rc()
    plt.figure()
    c = ["#EE6666", "#88BB44", "#EECC55"]
    for i, sim_df in enumerate(sim_dfs):
        error = np.sqrt(sim_df[r"e"] ** 2 + sim_df[r"h"] ** 2)
        plt.plot(sim_df["Time"], error, linewidth=4, color=c[i])
    plt.xlabel(xlabel="Time [s]", fontsize=12)
    plt.ylabel(ylabel="Tracking Error [m]", fontsize=12)
    # plt.ylim([0,15])
    plt.legend(
        [r"$\lambda_r=0.9$", r"$\lambda_r=0.5$", r"$\lambda_r=0.1$"],
        loc="upper right",
        fontsize=14,
    )
    # plt.show()
    path = os.path.join(output_dir, "control_errors.png")
    plt.savefig(path)
    if tracker:
        tracker.log_artifact(path)


def plot_3d(env, sim_df, output_dir, tracker=None):
    """
    Plots the AUV path in 3D inside the environment provided.
    """
    plt.figure()
    plt.rcdefaults()
    plt.rc("lines", linewidth=3)
    ax = env.plot3D()  # (wps_on=False)
    ax.plot3D(
        sim_df[r"$N$"],
        sim_df[r"$E$"],
        sim_df[r"$D$"],
        color="#EECC55",
        label="AUV Path",
    )  # , linestyle="dashed")
    ax.set_xlabel(xlabel="North [m]", fontsize=14)
    ax.set_ylabel(ylabel="East [m]", fontsize=14)
    ax.set_zlabel(zlabel="Down [m]", fontsize=14)
    ax.legend(loc="upper right", fontsize=14)
    # plt.show()
    path = os.path.join(output_dir, "trajectory_3d.png")
    plt.savefig(path)
    if tracker:
        tracker.log_artifact(path)


def plot_multiple_3d(env, sim_dfs):
    """
    Plots multiple AUV paths in 3D inside the environment provided.
    """
    plt.rcdefaults()
    c = ["#EE6666", "#88BB44", "#EECC55"]
    styles = ["dashed", "dashed", "dashed"]
    plt.rc("lines", linewidth=3)
    ax = env.plot3D()  # (wps_on=False)
    for i, sim_df in enumerate(sim_dfs):
        ax.plot3D(
            sim_df[r"$N$"],
            sim_df[r"$E$"],
            sim_df[r"$D$"],
            color=c[i],
            linestyle=styles[i],
        )
    ax.set_xlabel(xlabel="North [m]", fontsize=14)
    ax.set_ylabel(ylabel="East [m]", fontsize=14)
    ax.set_zlabel(zlabel="Down [m]", fontsize=14)
    ax.legend(
        ["Path", r"$\lambda_r=0.9$", r"$\lambda_r=0.5$", r"$\lambda_r=0.1$"],
        loc="upper right",
        fontsize=14,
    )
    plt.show()


def plot_current_data(sim_df, output_dir, tracker=None):
    plt.figure()
    set_default_plot_rc()
    # ---------------Plot current intensity------------------------------------
    current_labels = [r"$u_c$", r"$v_c$", r"$w_c$"]
    ax1 = sim_df.plot(x="Time", y=current_labels, linewidth=4, style=["-", "-", "-"])
    ax1.set_title("Current", fontsize=18)
    ax1.set_xlabel(xlabel="Time [s]", fontsize=14)
    ax1.set_ylabel(ylabel="Velocity [m/s]", fontsize=14)
    ax1.set_ylim([-1.25, 1.25])
    ax1.legend(loc="right", fontsize=14)
    # ax1.grid(color='k', linestyle='-', linewidth=0.1)
    # plt.show()
    path = os.path.join(output_dir, "currents.png")
    plt.savefig(path)
    if tracker:
        tracker.log_artifact(path)

    # ---------------Plot current direction------------------------------------
    """
    ax2 = ax1.twinx()
    ax2 = sim_df.plot(x="Time", y=[r"$\alpha_c$", r"$\beta_c$"], linewidth=4, style=["-", "--"] )
    ax2.set_title("Current", fontsize=18)
    ax2.set_xlabel(xlabel="Time [s]", fontsize=12)
    ax2.set_ylabel(ylabel="Direction [rad]", fontsize=12)
    ax2.set_ylim([-np.pi, np.pi])
    ax2.legend(loc="right", fontsize=12)
    ax2.grid(color='k', linestyle='-', linewidth=0.1)
    plt.show()
    """


def plot_collision_reward_function():
    horizontal_angles = np.linspace(-70, 70, 300)
    vertical_angles = np.linspace(-70, 70, 300)
    gamma_x = 25
    epsilon = 0.05
    sensor_readings = 0.4 * np.ones((300, 300))
    image = np.zeros((len(vertical_angles), len(horizontal_angles)))
    for i, horizontal_angle in enumerate(horizontal_angles):
        horizontal_factor = 1 - (abs(horizontal_angle) / horizontal_angles[-1])
        for j, vertical_angle in enumerate(vertical_angles):
            vertical_factor = 1 - (abs(vertical_angle) / vertical_angles[-1])
            beta = horizontal_factor * vertical_factor + epsilon
            image[j, i] = beta * (1 / (gamma_x * (sensor_readings[j, i]) ** 4))
    print(image.round(2))
    ax = plt.axes()
    plt.colorbar(plt.imshow(image), ax=ax)
    ax.imshow(image, extent=[-70, 70, -70, 70])
    ax.set_ylabel("Vertical vessel-relative sensor angle [deg]", fontsize=14)
    ax.set_xlabel("Horizontal vessel-relative sensor angle [deg]", fontsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.show()


if __name__ == "__main__":
    plot_collision_reward_function()
