import numpy as np
import parameters as p
import time


class HermiteTrajectory:
    def __init__(self, p0, v0, p1, v1, t_hit, t_start=0.0):
        self.p0 = np.asarray(p0, dtype=float)
        self.v0 = np.asarray(v0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)
        self.v1 = np.asarray(v1, dtype=float)
        self.t_hit = float(t_hit)
        self.t_start = float(t_start)

        # Precompute scaled tangents
        self.m0 = self.v0 * self.t_hit
        self.m1 = self.v1 * self.t_hit

    def get_state(self, t):
        elapsed = t - self.t_start
        s = np.clip(elapsed / self.t_hit, 0.0, 1.2)

        s2 = s*s
        s3 = s2*s

        h00 =  2*s3 - 3*s2 + 1
        h10 =      s3 - 2*s2 + s
        h01 = -2*s3 + 3*s2
        h11 =      s3 -     s2

        # Position (x,y)
        pos_xy = (
            h00 * self.p0[:2] +
            h10 * self.m0[:2] +
            h01 * self.p1[:2] +
            h11 * self.m1[:2]
        )

        # Orientation (theta)
        theta0 = self.p0[2]
        theta1 = self.p1[2]
        dtheta = wrap_angle(theta1 - theta0)

        theta = (
            h00 * theta0 +
            h10 * self.m0[2] +
            h01 * (theta0 + dtheta) +
            h11 * self.m1[2]
        )

        theta = wrap_angle(theta)

        return np.array([pos_xy[0], pos_xy[1], theta])
    
    def get_velocity(self, t: float) -> np.ndarray:
        """
        Returns the velocity [vx, vy, v_theta] at time t.
        Returns zeros if outside the time interval.
        """
        elapsed = t - self.t_start
        if elapsed < 0 or elapsed > self.t_hit:
            return np.zeros(3)

        s = elapsed / self.t_hit
        ds_dt = 1.0 / self.t_hit if self.t_hit > 1e-9 else 0.0

        s2 = s * s
        s3 = s2 * s

        # Derivatives of basis functions w.r.t. s
        dh00 =  6*s2 - 6*s
        dh10 =  3*s2 - 4*s + 1
        dh01 = -6*s2 + 6*s
        dh11 =  3*s2 - 2*s

        # For position
        vel_xy = ds_dt * (
            dh00 * self.p0[:2] +
            dh10 * self.m0[:2] +
            dh01 * self.p1[:2] +
            dh11 * self.m1[:2]
        )

        # For theta
        theta0 = self.p0[2]
        dtheta = wrap_angle(self.p1[2] - theta0)

        vel_theta = ds_dt * (
            dh00 * theta0 +
            dh10 * self.m0[2] +
            dh01 * (theta0 + dtheta) +
            dh11 * self.m1[2]
        )

        return np.array([vel_xy[0], vel_xy[1], vel_theta])

def wrap_angle(theta):
    return (theta + np.pi) % (2*np.pi) - np.pi


def compute_hit_velocity(p_hit, k_hit=0.4):
    x_hit = p_hit[0]
    x_target = 0 if x_hit < 0 else -0
    
    v_x = (x_target - x_hit)/p.T_flight
    v_y = 0.5*p.g*p.T_flight

    v = np.array([v_x, v_y])
    theta = np.atan2(v_y, v_x) - np.pi/2

    return k_hit * v, theta


def is_trajectory_inside_workspace(traj, num_samples=30):
    #Sample the trajectory and check if all points are inside bounds.
    duration = traj.t_hit
    times = np.linspace(0, duration, num_samples)
    for dt in times:
        pos = traj.get_state(traj.t_start + dt)  # get [x, y, theta]
        x, y = pos[:2]
        if not (p.WORKSPACE_X_MIN <= x <= p.WORKSPACE_X_MAX and
                p.WORKSPACE_Y_MIN <= y <= p.WORKSPACE_Y_MAX):
            return False
    return True



def evaluate_trajectory(trajectory, t):
    # Unified way to get position at time t, whether single or chained.
    if trajectory is None:
        return np.array([0.0, 0.0, 0.0])  # fallback

    if isinstance(trajectory, dict) and trajectory.get('type') == 'chained':
        to_wait = trajectory['to_wait']
        wait_start = to_wait.t_start + to_wait.t_hit   # end of first segment
        from_wait = trajectory['from_wait']

        if t < wait_start:
            return to_wait.get_state(t)
        elif t < from_wait.t_start:
            # during wait phase: hold at wait point (p1 of to_wait)
            return to_wait.p1.copy()
        else:
            return from_wait.get_state(t)
    else:
        # normal single Hermite trajectory
        return trajectory.get_state(t)
    

def evaluate_velocity(trajectory, t: float) -> np.ndarray:
    """
    Unified way to get velocity at time t, whether single HermiteTrajectory
    or chained (dict) structure.
    Returns zeros if no valid trajectory or outside time range.
    """
    if trajectory is None:
        return np.zeros(3)

    if isinstance(trajectory, dict) and trajectory.get('type') == 'chained':
        to_wait = trajectory['to_wait']
        wait_start = to_wait.t_start + to_wait.t_hit
        from_wait = trajectory['from_wait']

        if t < wait_start:
            return to_wait.get_velocity(t)
        elif t < from_wait.t_start:
            return np.zeros(3)
        else:
            return from_wait.get_velocity(t)
    else:
        try:
            return trajectory.get_velocity(t)
        except AttributeError:
            return np.zeros(3)


def voltage_regulator(odrv, d_des, V_THRESH=24.5, BIAS_MAX=0.02, K_V=0.005):
    Vbus = odrv.vbus_voltage
    if Vbus > V_THRESH:
        delta_d = min(K_V * (Vbus - V_THRESH), BIAS_MAX)
    else:
        delta_d = 0.0

    d_des -= delta_d
    return d_des