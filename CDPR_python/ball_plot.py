import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- CONFIG ----

T_START = 11.61
T_WINDOW = 30      # None = full log

mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.figsize": (8, 5),
    "lines.linewidth": 1.5,
    "axes.grid": True,
})

# ---- HELPERS ----

def crop_time_window(t, data_dict, t_start, t_window):

    if t_window is None:
        return t, data_dict

    t_end = t_start + t_window

    t_rel = t - t[0]

    mask = (
        (t_rel >= t_start)
        &
        (t_rel <= t_end)
    )

    t_new = t[mask]

    data_new = {}

    for key, val in data_dict.items():

        if val is None:

            data_new[key] = None

        elif isinstance(val, np.ndarray):

            if len(val) == len(t):

                data_new[key] = val[mask]

            else:

                idx_map = np.linspace(
                    0,
                    len(val)-1,
                    len(t)
                ).astype(int)

                data_new[key] = val[idx_map][mask]

        else:

            data_new[key] = val

    return t_new, data_new


# ---- PLOTTER ----

class BallPlotter:

    def __init__(self, data):

        # ---- LOAD DATA ----

        t = data["t"]

        data_dict = {
            "ball_pos": data["ball_pos"],
            "ball_vel": data["ball_vel"],

            "pos_ref": data.get("ref_pos", None),
            "vel_ref": data.get("ref_vel", None),

            "pos_raw": data["pos_raw"],
            "vel_raw": data["vel_raw"],

            "p_hit": data.get("p_hit", None),
            "q": data.get("q", None),
        }

        t, cropped = crop_time_window(
            t,
            data_dict,
            T_START,
            T_WINDOW
        )

        self.t = t - t[0]

        self.ball_pos = cropped["ball_pos"]
        self.ball_vel = cropped["ball_vel"]

        self.pos_ref = cropped.get("pos_ref", None)
        self.vel_ref = cropped.get("vel_ref", None)

        self.pos_raw = cropped["pos_raw"]
        self.vel_raw = cropped["vel_raw"]

        self.p_hit = cropped.get("p_hit", None)
        self.q = cropped.get("q", None)

        # ---- CREATE FIGURES ----

        self.create_xy_plot()

        self.create_position_plot()

        self.create_velocity_plot()

        self.create_speed_plot()

        self.create_hit_prediction_plot()

    # ---- XY TRAJECTORY ----

    def create_xy_plot(self):

        fig, ax = plt.subplots()

        if self.pos_ref is not None:

            ax.plot(
                self.pos_ref[:, 0],
                self.pos_ref[:, 1],
                "--",
                label="desired trajectory"
            )

        ax.plot(
            self.ball_pos[:, 0],
            self.ball_pos[:, 1],
            label="measured trajectory"
        )

        ax.set_title("Ball trajectory (XY)")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        ax.axis("equal")

    # ---- POSITION VS TIME ----

    def create_position_plot(self):

        fig, ax = plt.subplots()

        """ax.plot(
            self.t,
            self.pos_raw[:, 0],
            "--",
            label="x raw"
        )"""

        ax.plot(
            self.t,
            self.ball_pos[:, 0],
            label="x"
        )

        if self.pos_ref is not None:

            ax.plot(
                self.t,
                self.pos_ref[:, 0],
                "--",
                label="x des"
            )

            ax.plot(
                self.t,
                self.pos_ref[:, 1],
                "--",
                label="y des"
            )

        ax.plot(
            self.t,
            self.pos_raw[:, 1],
            "--",
            label="y raw"
        )

        ax.plot(
            self.t,
            self.ball_pos[:, 1],
            label="y"
        )

        ax.set_title("Ball position vs time")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")

        # ---- TRACKING ERROR METRICS ----

        if self.pos_ref is not None:

            # position error vector
            e = self.ball_pos - self.pos_ref[:, :2]

            # RMS error
            rms = np.sqrt(np.mean(e**2))

            # peak absolute error
            peak = np.max(np.abs(e))

            # axis-wise RMS
            rms_x = np.sqrt(np.mean(e[:, 0]**2))
            rms_y = np.sqrt(np.mean(e[:, 1]**2))

            # axis-wise peak
            peak_x = np.max(np.abs(e[:, 0]))
            peak_y = np.max(np.abs(e[:, 1]))

            text = (
                f"RMS: {rms*1000:.1f} mm\n"
                f"Peak: {peak*1000:.1f} mm\n\n"
                f"RMS x: {rms_x*1000:.1f} mm\n"
                f"RMS y: {rms_y*1000:.1f} mm\n\n"
                f"Peak x: {peak_x*1000:.1f} mm\n"
                f"Peak y: {peak_y*1000:.1f} mm"
            )

            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
            )

            print("\n=== Ball tracking performance ===")
            print(f"RMS error:  {rms*1000:.2f} mm")
            print(f"Peak error: {peak*1000:.2f} mm")

        ax.legend()

    # ---- VELOCITY VS TIME ----

    def create_velocity_plot(self):

        fig, ax = plt.subplots()

        ax.plot(
            self.t,
            self.vel_raw[:, 0], 
            "--",
            label="vx raw"
        )

        ax.plot(
            self.t,
            self.ball_vel[:, 0],
            label="vx"
        )
        
        if self.vel_ref is not None:

            ax.plot(
                self.t,
                self.vel_ref[:, 0],
                "--",
                label="vx des"
            )

            ax.plot(
                self.t,
                self.vel_ref[:, 1],
                "--",
                label="vy des"
            )

        ax.plot(
            self.t,
            self.vel_raw[:, 1],
            "--",
            label="vy raw"
        )

        ax.plot(
            self.t,
            self.ball_vel[:, 1],
            label="vy"
        )


        ax.set_title("Ball velocity vs time")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity [m/s]")

        ax.legend()

    # ---- SPEED MAGNITUDE ----

    def create_speed_plot(self):

        fig, ax = plt.subplots()

        speed = np.linalg.norm(
            self.ball_vel,
            axis=1
        )

        ax.plot(
            self.t,
            speed
        )

        ax.set_title("Ball speed")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Speed [m/s]")

    # ---- HIT PREDICTION ----

    def create_hit_prediction_plot(self):

        if self.p_hit is None:
            return

        fig, ax = plt.subplots()

        valid = ~np.isnan(self.p_hit[:, 0])

        ax.plot(
            self.t[valid],
            self.p_hit[valid, 0],
            label="Predicted hit x"
        )

        ax.plot(
            self.t[valid],
            self.p_hit[valid, 1],
            label="Predicted hit y"
        )

        ax.set_title("Predicted impact position")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Position [m]")

        ax.legend()


# ---- MAIN ----

def run(filepath):

    data = np.load(filepath, allow_pickle=True)

    BallPlotter(data)

    plt.show()


if __name__ == "__main__":

    run("logs/ball_manipulation/balance_step_.npz")