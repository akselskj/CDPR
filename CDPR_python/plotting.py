import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib as mpl
import time

T_START = 1
T_WINDOW = 3  # seconds to keep (set to None to disable)

mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "lines.linewidth": 1.5,
    "axes.grid": True
})

def compute_d_dot(d, t):
    d_dot = np.zeros_like(d)
    for i in range(d.shape[1]):
        d_dot[:, i] = np.gradient(d[:, i], t)
    return d_dot

def compensate_delay(d, d_dot, tau):
    tau = np.array(tau)
    return d + d_dot * tau

def estimate_delay(d, d_des, d_dot):
    tau_est = np.zeros_like(d)
    for i in range(d.shape[1]):
        for k in range(len(d)):
            if abs(d_dot[k, i]) > 1e-4:
                tau_est[k, i] = (d_des[k, i] - d[k, i]) / d_dot[k, i]
            else:
                tau_est[k, i] = np.nan
    return tau_est

def crop_time_window(t, data_dict, t_start, t_window):

    if t_window is None:
        return t, data_dict

    t_end = t_start + t_window

    # mask on reference time
    mask = (t >= t_start) & (t <= t_end)

    t_new = t[mask]

    data_new = {}

    for key, val in data_dict.items():

        if val is None:
            data_new[key] = None

        elif isinstance(val, np.ndarray):

            # case 1: same length → direct mask
            if len(val) == len(t):
                data_new[key] = val[mask]

            # case 2: different length → resample indices
            else:
                # map original t indices to this signal
                idx_map = np.linspace(0, len(val)-1, len(t)).astype(int)
                data_new[key] = val[idx_map][mask]

        else:
            data_new[key] = val

    return t_new, data_new


def compute_total_rms_error(d, d_des, d_dot, tau):
    # delay compensation
    d_comp = compensate_delay(d, d_dot, tau)

    # error (in meters)
    e = d_des - d_comp

    # flatten all cables into one vector
    e_flat = e.reshape(-1)

    # RMS
    rms = np.sqrt(np.mean(e_flat**2))

    return rms

class Plotter:

    def __init__(self, data):

        t = data["t"]

        data_dict = {
            "q_des": data["q_des"],
            "q_est": data["q_est"],
            "d": data["d"],
            "d_des": data["d_des"],
            "voltage": data["voltage"],
            "force": data.get("force", None),
            "dt": data.get("dt"),
            "read_time": data.get("read_time"),
            "send_time": data.get("send_time"),
            "compute_time": data.get("compute_time"),
        }

        self.t_full = data["t"]
        self.d_full = data["d"]
        self.d_des_full = data["d_des"]
        self.d_dot_full = compute_d_dot(self.d_full, self.t_full)

        t, cropped = crop_time_window(t, data_dict, T_START, T_WINDOW)

        self.t = t
        self.t = self.t - self.t[0]

        self.q_des = cropped["q_des"]
        self.q_est = cropped["q_est"]
        self.d = cropped["d"]
        self.d_des = cropped["d_des"]
        self.voltage = cropped["voltage"]
        self.force = cropped.get("force", None)
        self.dt = cropped["dt"]
        self.read_time = cropped.get("read_time", None)
        self.send_time = cropped.get("send_time", None)
        self.compute_time = cropped.get("compute_time", None)

        self.d_dot = compute_d_dot(self.d, self.t)
        self.tau = [0.008]*4

        self.create_static_figures()
        self.create_error_figure()
        self.create_sliders()

        self.update_error_plot()
        
        self.update_pending = False

        self.timer = self.fig_err.canvas.new_timer(interval=10)
        self.timer.add_callback(self.periodic_update)
        self.timer.start()

    # ---- STATIC FIGURES ----
    def create_static_figures(self):

        # XY
        fig, ax = plt.subplots()
        ax.plot(self.q_des[:,0], self.q_des[:,1], "b--")
        ax.plot(self.q_est[:,0], self.q_est[:,1], "r")
        ax.set_title("Platform trajectory (XY)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.legend(["Desired", "Estimated"])

        # Position
        fig, ax = plt.subplots()
        ax.plot(self.t, self.q_des[:,0], "r--")
        ax.plot(self.t, self.q_est[:,0], "r")
        ax.plot(self.t, self.q_des[:,1], "b--")
        ax.plot(self.t, self.q_est[:,1], "b")
        ax.set_title("Platform position vs time")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")
        ax.legend(["x_des", "x_est", "y_des", "y_est"])

        # Theta
        fig, ax = plt.subplots()
        ax.plot(self.t, self.q_des[:,2]*180/np.pi, "k--")
        ax.plot(self.t, self.q_est[:,2]*180/np.pi, "k")
        ax.set_title("Platform orientation")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Angle [deg]")
        ax.legend(["θ_des", "θ_est"])

        # Cable length
        fig, ax = plt.subplots()
        for i in range(4):
            ax.plot(self.t, self.d[:,i])
            ax.plot(self.t, self.d_des[:,i], "--")
            ax.set_title("Cable lengths")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Length [m]")
        labels = []
        for i in range(4):
            labels += [f"d{i}", f"d{i}_des"]
        ax.legend(labels)

        # dt
        fig, ax = plt.subplots()

        ax.plot(self.t, self.dt[:], "b")

        # average dt
        avg_dt = np.mean(self.dt)
        avg_freq = 1 / avg_dt

        # horizontal line for average
        ax.axhline(avg_dt, color="r", linestyle="--",
                label=f"Avg dt = {avg_dt*1000:.1f} ms ({avg_freq:.1f} Hz)")

        ax.set_title("Loop dt")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Loop time [s]")

        ax.legend()

                # Timing breakdown
        if self.read_time is not None:

            fig, ax = plt.subplots()

            ax.plot(self.t, self.read_time * 1000, label="Read")
            ax.plot(self.t, self.send_time * 1000, label="Write")
            ax.plot(self.t, self.compute_time * 1000, label="Compute")

            ax.set_title("Control loop timing breakdown")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Time [ms]")

            ax.legend()

            # ---- print averages ----
            print("\n=== Average timing ===")
            print(f"Read:    {np.mean(self.read_time)*1000:.3f} ms")
            print(f"Write:   {np.mean(self.send_time)*1000:.3f} ms")
            print(f"Compute: {np.mean(self.compute_time)*1000:.3f} ms")

            total = (
                np.mean(self.read_time)
                + np.mean(self.send_time)
                + np.mean(self.compute_time)
            )

            print(f"Total:   {total*1000:.3f} ms")
            print(f"Freq:    {1/total:.1f} Hz")
        

                    # ---- average timing bar plot ----
            fig, ax = plt.subplots()

            labels = ["Read", "Write", "Compute"]

            values = [
                np.mean(self.read_time) * 1000,
                np.mean(self.send_time) * 1000,
                np.mean(self.compute_time) * 1000,
            ]

            ax.bar(labels, values)

            ax.set_title("Average control loop timing distribution")
            ax.set_ylabel("Average time [ms]")

        # Bus voltage
        fig, ax = plt.subplots()
        for i in range(self.voltage.shape[1]):
            ax.plot(self.t, self.voltage[:, i], label=f"ODrive {i}")
        ax.set_title("Bus voltage")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Voltage [V]")
        ax.legend()


        # Force controlled cable
        if self.force is not None:
            fig, ax = plt.subplots()

            # Handle possible length mismatch (same issue as voltage)
            min_len = min(len(self.t), len(self.force))

            t_force = self.t[:min_len]
            force = self.force[:min_len]

            ax.plot(t_force, force, "m")

            ax.set_title("Active force-controlled cable")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Cable index")

            ax.set_yticks([0, 1, 2, 3])


        # Delay estimation
        fig, ax = plt.subplots()
        tau_est = estimate_delay(self.d_full, self.d_des_full, self.d_dot_full)
        def moving_average(x, N=20):
            return np.convolve(x, np.ones(N)/N, mode='same')

        for i in range(4):
            tau_smooth = moving_average(tau_est[:, i])
            ax.plot(self.t_full, tau_smooth, label=f"{i}")
        ax.set_ylim(0, 0.02)
        ax.legend()
        ax.set_title("Estimated delay")

    # ---- ERROR FIGURE (ONLY DYNAMIC) ----
    def create_error_figure(self):

        self.fig_err, self.ax_err = plt.subplots()

        self.lines_raw = []
        self.lines_comp = []

        for i in range(4):
            l1, = self.ax_err.plot([], [], ":", label=f"{i} raw")
            l2, = self.ax_err.plot([], [], label=f"{i} comp")
            self.lines_raw.append(l1)
            self.lines_comp.append(l2)
        
        """min_len = min(len(self.t), len(self.force))

        t_force = self.t[:min_len]
        force = self.force[:min_len]
    
        self.ax_err.plot(t_force, force, "m")"""

        self.rms_text = self.ax_err.text(
            0.02, 0.95, "", transform=self.ax_err.transAxes,
            verticalalignment='top'
        )
        
        self.ax_err.set_title("Cable Error")
        self.ax_err.set_title("Cable length error (raw vs delay compensated)")
        self.ax_err.set_xlabel("Time [s]")
        self.ax_err.set_ylabel("Error [cm]")
        self.ax_err.legend(ncol=2)

    def update_error_plot(self):

        d_comp = compensate_delay(self.d, self.d_dot, self.tau)

        # compute RMS (in meters)
        e = self.d_des - d_comp
        rms = np.sqrt(np.mean(e.reshape(-1)**2))

        # update plot lines
        for i in range(4):
            y_raw  = (self.d_des[:,i] - self.d[:,i]) * 100
            y_comp = (self.d_des[:,i] - d_comp[:,i]) * 100

            self.lines_raw[i].set_data(self.t, y_raw)
            self.lines_comp[i].set_data(self.t, y_comp)

        # update RMS display (convert to mm)
        max_err = np.max(np.abs(e))
        self.rms_text.set_text(f"RMS: {rms*1000:.2f} mm\nMax: {max_err*1000:.2f} mm")

        self.ax_err.relim()
        self.ax_err.autoscale_view()

        self.fig_err.canvas.draw_idle()
        self.fig_err.canvas.flush_events()

    # ---- SLIDERS ----
    def create_sliders(self):

        self.fig_slider = plt.figure("Sliders", figsize=(5,3))
        self.sliders = []

        for i in range(4):
            ax = self.fig_slider.add_axes([0.2, 0.7 - i*0.15, 0.6, 0.05])
            s = Slider(ax, f"τ{i}", 0.0, 0.02, valinit=self.tau[i])
            s.on_changed(self.update_tau)
            self.sliders.append(s)

    def update_tau(self, val):
        self.tau = [s.val for s in self.sliders]
        self.update_pending = True

    def periodic_update(self):
        if self.update_pending:
            self.update_error_plot()
            self.update_pending = False


def run(filepath):
    data = np.load(filepath)
    Plotter(data)
    plt.show()


if __name__ == "__main__":
    run("logs/controller_tests/dt_test_position_00_20260508_150615.npz")