import os
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

import parameters as p

"""
this file contains helper functions
"""


# MATH / KINEMATICS (put here as it is only useful for FDSI)

def structure_matrix(a, b, q):
    """
    Compute the structure matrix A_T for a planar CDPR.

    Parameters
    ----------
    a : ndarray, shape (2, 4)
        Frame anchor points
    b : ndarray, shape (2, 4)
        Platform anchor points (body frame)
    theta : float
        Platform orientation [rad]
    x, y : float
        Platform position

    Returns
    -------
    A_T : ndarray, shape (3, 4)
        Structure matrix (transposed Jacobian)
    """

    x = q[0]
    y = q[1]
    theta = q[2]

    # Rotation matrix R_BI
    R_BI = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    # Unit cable direction vectors
    u = np.zeros((2, 4))
    for i in range(4):
        ell_i = a[:, i] - np.array([x, y]) - R_BI @ b[:, i]
        u[:, i] = ell_i / np.linalg.norm(ell_i)

    # Platform anchor points in inertial frame
    bi = np.zeros((2, 4))
    for i in range(4):
        bi[:, i] = R_BI @ b[:, i]

    # z-component of (b_i × u_i)
    bcrossu = np.zeros(4)
    for i in range(4):
        bcrossu[i] = bi[0, i] * u[1, i] - bi[1, i] * u[0, i]

    # Structure matrix
    A_T = np.vstack((u, bcrossu))

    return A_T


def select_force_controlled_cables(a, b, q_des, j_force_current, gamma):
    """
    SELECT_FORCE_CONTROLLED_CABLES

    Computes FD-sensitivity phi for each candidate force-controlled cable
    and selects the best one using hysteresis switching.

    Parameters
    ----------
    a : ndarray, shape (2, 4)
        Frame anchor points
    b : ndarray, shape (2, 4)
        Platform anchor points
    q_des : array-like, shape (3,)
        Desired pose [x, y, theta]
    j_force_current : int
        Currently force-controlled cable index (0-based)
    gamma : float
        Hysteresis tolerance factor (< 1 recommended, e.g. 0.995)

    Returns
    -------
    j_force_new : int
        Updated force-controlled cable index (0-based)
    sigma_min : float
        Minimum FD sensitivity value
    """

    n = 4

    # Candidate force-controlled cables (0-based!)
    comb_list = [0, 1, 2, 3]

    # Structure matrix at desired pose
    A_T = structure_matrix(a, b, q_des)

    sigma_vals = np.full(len(comb_list), np.inf)

    # --- Evaluate σ for all candidates ---
    for ci, jset in enumerate(comb_list):
        cols_c = [jset]
        cols_d = [i for i in range(n) if i not in cols_c]

        if len(cols_d) == 0:
            continue

        S = -np.linalg.pinv(A_T[:, cols_d]) @ A_T[:, cols_c]
        sigma_vals[ci] = np.linalg.norm(S, ord=np.inf)

    # Best candidate
    best_idx = int(np.argmin(sigma_vals))
    sigma_min = sigma_vals[best_idx]

    # --- Evaluate current choice ---
    cols_c_cur = [j_force_current]
    cols_d_cur = [i for i in range(n) if i not in cols_c_cur]

    if len(cols_d_cur) == 0 or len(cols_c_cur) == 0:
        sigma_cur = np.inf
    else:
        S_cur = -np.linalg.pinv(A_T[:, cols_d_cur]) @ A_T[:, cols_c_cur]
        sigma_cur = np.linalg.norm(S_cur, ord=np.inf)

    # --- Apply hysteresis ---
    if sigma_min < gamma * sigma_cur:
        j_force_new = best_idx
    else:
        j_force_new = j_force_current

    return j_force_new, sigma_min


def rot(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])


def cable_length_from_encoder(phi, phi0, i):
    return p.motor_signs[i] * (phi0[i] - phi) * 2*np.pi*p.r_d


def encoder_from_cable_length(d, phi0, i):
    return p.motor_signs[i] * d / (2*np.pi*p.r_d) + phi0[i]


# LOGGING

def save_experiment_data(filename_prefix,
                         t_log,
                         d_des_log,
                         d_log,
                         q_des_log,
                         q_est_log,
                         torque_log,
                         voltage_log,
                         force_log=None,
                         dt_log=None,
                         read_log=None,
                         send_log=None,
                         compute_log=None):

    # ---- convert to numpy ----
    data = {
        "t": np.array(t_log),
        "d_des": np.array(d_des_log),
        "d": np.array(d_log),
        "q_des": np.array(q_des_log),
        "q_est": np.array(q_est_log),
        "torque": np.array(torque_log),
        "voltage": np.array(voltage_log),
    }

    # ---- optional logs ----
    if force_log is not None:
        data["force"] = np.array(force_log)

    if dt_log is not None:
        data["dt"] = np.array(dt_log)

    if read_log is not None:
        data["read_time"] = np.array(read_log)

    if send_log is not None:
        data["send_time"] = np.array(send_log)

    if compute_log is not None:
        data["compute_time"] = np.array(compute_log)

    # ---- create filename ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.npz"

    # ---- optional: save in folder ----
    os.makedirs("logs", exist_ok=True)
    filepath = os.path.join("logs", filename)

    # ---- save ----
    np.savez_compressed(filepath, **data)

    print(f"Saved log to: {filepath}")


# PLOTTING


def plot_trajectory(t_log, d_des_log, d_log, q_des_log, q_est_log,
                    torque_log, voltage_log, force_log=None,
                    dt_log=None, exec_dt_log=None):

    # ---- Convert to arrays ----
    q_des_log = np.array(q_des_log)
    q_est_log = np.array(q_est_log)
    d_des_log = np.array(d_des_log)
    d_log = np.array(d_log)
    torque_log = np.array(torque_log)
    voltage_log = np.array(voltage_log)
    t_log = np.array(t_log)

    # ---- Trim to shortest length ----
    min_len = min(len(t_log), len(q_des_log), len(q_est_log),
                  len(d_log), len(d_des_log),
                  len(torque_log), len(voltage_log))

    t_log = t_log[:min_len]
    d_log = d_log[:min_len]
    d_des_log = d_des_log[:min_len]
    q_des_log = q_des_log[:min_len]
    q_est_log = q_est_log[:min_len]
    torque_log = torque_log[:min_len]
    voltage_log = voltage_log[:min_len]

    # ---- Delay compensation ----
    tau = [0.0085, 0.0085, 0.0085, 0.006]  # seconds

    d_dot = np.zeros_like(d_log)
    d_log_comp = np.zeros_like(d_log)

    for i in range(4):
        d_dot[:, i] = np.gradient(d_log[:, i], t_log)

        d_log_comp[:, i] = d_log[:, i] + d_dot[:, i] * tau[i]

    # ---- estimate time delay per cable ----
    tau_est = np.zeros_like(d_log)

    velocity_threshold = 1e-3  # m/s (tune if needed)

    for i in range(4):
        for k in range(len(t_log)):
            if abs(d_dot[k, i]) > velocity_threshold:
                tau_est[k, i] = (d_des_log[k, i] - d_log[k, i]) / d_dot[k, i]
            else:
                tau_est[k, i] = np.nan  # ignore unreliable values
    from scipy.signal import savgol_filter

    for i in range(4):
        valid = ~np.isnan(tau_est[:, i])
        if np.sum(valid) > 10:
            tau_est[valid, i] = savgol_filter(tau_est[valid, i], 21, 2)

    # =========================
    # XY plot
    # =========================
    plt.figure()
    plt.plot(q_des_log[:,0], q_des_log[:,1], "b--", label="desired")
    plt.plot(q_est_log[:,0], q_est_log[:,1], "r", label="estimated")
    plt.title("XY Pose")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.grid()

    # =========================
    # Position vs time
    # =========================
    plt.figure()
    plt.plot(t_log, q_des_log[:,0], "r--", label="x desired")
    plt.plot(t_log, q_est_log[:,0], "r", label="x estimated")
    plt.plot(t_log, q_des_log[:,1], "b--", label="y desired")
    plt.plot(t_log, q_est_log[:,1], "b", label="y estimated")
    plt.title("Position vs Time")
    plt.xlabel("time [s]")
    plt.legend()
    plt.grid()

    # =========================
    # Orientation
    # =========================
    plt.figure()
    plt.plot(t_log, q_des_log[:,2]*180/np.pi, "k--", label="theta desired")
    plt.plot(t_log, q_est_log[:,2]*180/np.pi, "k", label="theta estimated")
    plt.title("Orientation")
    plt.xlabel("time [s]")
    plt.ylabel("deg")
    plt.legend()
    plt.grid()

    # =========================
    # Cable lengths
    # =========================
    plt.figure()
    for i in range(4):
        plt.plot(t_log, d_log[:,i], label=f"Cable {i}")
        plt.plot(t_log, d_des_log[:,i], "--", label=f"Cable {i} desired")
    plt.title("Cable Lengths")
    plt.xlabel("time [s]")
    plt.ylabel("Length [m]")
    plt.legend()
    plt.grid()

    # =========================
    # Estimated delay over time
    # =========================
    plt.figure()
    for i in range(4):
        plt.plot(t_log, tau_est[:, i], label=f"Cable {i}")

    plt.axhline(0.01, linestyle="--", color="k", alpha=0.5, label="reference τ")

    plt.title("Estimated Time Delay per Cable")
    plt.xlabel("time [s]")
    plt.ylabel("Delay [s]")
    plt.ylim(0, 0.02)  # adjust if needed
    plt.legend()
    plt.grid()

    # =========================
    # Cable length error (RAW + COMPENSATED)
    # =========================
    plt.figure()
    for i in range(4):
        # raw
        plt.plot(t_log, (d_des_log[:,i] - d_log[:,i])*100,
                 "--", alpha=0.5, label=f"Cable {i} raw")

        # compensated
        plt.plot(t_log, (d_des_log[:,i] - d_log_comp[:,i])*100,
                 label=f"Cable {i} compensated")

    plt.title("Cable Length Error (raw vs compensated)")
    plt.xlabel("time [s]")
    plt.ylabel("Length [cm]")
    plt.legend()
    plt.grid()

    # =========================
    # Torque
    # =========================
    plt.figure()
    for i in range(4):
        plt.plot(t_log, torque_log[:,i], label=f"Cable {i}")
    plt.title("Cable Torque")
    plt.xlabel("time [s]")
    plt.ylabel("Torque [Nm]")
    plt.legend()
    plt.grid()

    # =========================
    # Voltage
    # =========================
    plt.figure()
    for i in range(4):
        plt.plot(t_log, voltage_log[:,i], label=f"ODrive {i}")
    plt.title("Bus Voltage")
    plt.xlabel("time [s]")
    plt.ylabel("Voltage [V]")
    plt.legend()
    plt.grid()

    # =========================
    # Optional plots
    # =========================
    if force_log is not None:
        force_log = np.array(force_log)
        plt.figure()
        plt.plot(t_log[:len(force_log)], force_log, "m")
        plt.title("Force-controlled cable (σ)")
        plt.xlabel("time [s]")
        plt.grid()

    if dt_log is not None:
        dt_log = np.array(dt_log)
        plt.figure()
        plt.plot(t_log[:len(dt_log)], dt_log, "c")
        plt.title("Loop dt")
        plt.xlabel("time [s]")
        plt.ylabel("seconds")
        plt.grid()

    if exec_dt_log is not None:
        exec_dt_log = np.array(exec_dt_log)
        plt.figure()
        plt.plot(exec_dt_log, "c")
        plt.title("Executor loop dt")
        plt.ylabel("seconds")
        plt.grid()

    plt.show()


_live_plot = {
    "fig": None,
    "ax": None,
    "platform": None,
    "cables": None,
    "text_pose": None,
    "text_lenghts": None,
    "initialized": False,
}


def init_live_cdpr_plot():
    """
    Initialize a non-blocking live plot for CDPR visualization.
    Call ONCE before the control loop starts.
    """

    if _live_plot["initialized"]:
        return

    plt.ion()  # interactive mode ON

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_aspect("equal")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_ylabel("y [m]")
    ax.set_xlabel("x [m]")
    ax.set_title("CDPR – Live Pose Estimate")
    ax.grid(True, alpha=0.3)

    a = np.array(p.a)  # frame anchors
    b = np.array(p.b)  # platform anchors (body frame)

    # frame anchors
    ax.plot(a[0, :], a[1, :], "ks", label="Frame")

    # --- motor labels ---
    for i in range(4):
        ax.text(
            a[0, i] + 0.03,   # small offset so it doesn't overlap marker
            a[1, i] + 0.03,
            f"M{i}",
            fontsize=12,
            color="black",
            weight="bold"
        )

    # platform outline
    platform, = ax.plot([], [], "ro-", lw=2, label="Platform")

    # cables
    cables = []
    for _ in range(4):
        line, = ax.plot([], [], "b-", lw=1)
        cables.append(line)

    # text boxes
    text_pose = ax.text(
        1.03, 0.95, "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    text_lengths = ax.text(
        1.03, 0.82, "",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    )

    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()

    _live_plot.update({
        "fig": fig,
        "ax": ax,
        "platform": platform,
        "cables": cables,
        "text_pose": text_pose,
        "text_lenghts": text_lengths,
        "initialized": True,
    })


def update_live_cdpr_plot(q, d_lengths):
    """
    Update the live CDPR plot.
    Call ONCE per control-loop iteration.

    Parameters
    ----------
    q : array-like (3,)
        Estimated platform pose [x, y, theta]
    """

    if not _live_plot["initialized"]:
        init_live_cdpr_plot()

    fig = _live_plot["fig"]
    ax = _live_plot["ax"]
    platform = _live_plot["platform"]
    cables = _live_plot["cables"]
    text_pose = _live_plot["text_pose"]
    text_lengths = _live_plot["text_lenghts"]

    a = np.array(p.a)
    b = np.array(p.b)

    q = np.asarray(q)
    x, y, theta = q
    theta_deg = np.degrees(theta)

    R = rot(q[2])
    r = q[:2]

    # platform anchor points in world frame
    C = r.reshape(2, 1) + R @ b

    # update platform polygon
    platform.set_data(
        np.append(C[0, :], C[0, 0]),
        np.append(C[1, :], C[1, 0])
    )

    # update cables
    for i in range(4):
        cables[i].set_data(
            [a[0, i], C[0, i]],
            [a[1, i], C[1, i]]
        )

    # Update pose text (top-left)
    pose_str = (
        f"Platform Pose:\n"
        f"  x = {x:.3f} m\n"
        f"  y = {y:.3f} m\n"
        f"  θ = {theta_deg:.1f}° ({theta:.3f} rad)"
    )
    text_pose.set_text(pose_str)

    # Update cable lengths text (below pose)
    if d_lengths is not None:
        d_lengths = np.asarray(d_lengths)
        lengths_str = "Cable Lengths [m]:\n"
        for i in range(4):
            lengths_str += f"  Cable {i}: {d_lengths[i]:.4f}\n"
        text_lengths.set_text(lengths_str)
    else:
        text_lengths.set_text("Cable lengths: not provided")

    # ---- NON-BLOCKING DRAW ----
    fig.canvas.draw_idle()
    plt.pause(0.001)
