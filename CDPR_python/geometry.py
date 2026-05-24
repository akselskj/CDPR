import time
import numpy as np
import parameters as p
import utils as utils
import trajectory_planner as traj
import keyboard
import mouse


def cable_jacobian(q, a, b):
    """
    Compute the 4×3 Jacobian matrix J such that
        cable_vel = - J @ q_dot
    (negative sign because shortening cable pulls platform)

    Parameters
    ----------
    q : array (3,)     [x, y, theta]
    a : array (2,4)    fixed anchors
    b : array (2,4)    body anchors

    Returns
    -------
    J : array (4,3)    ∂d_i / ∂q_j
    """
    q = np.asarray(q)
    a = np.asarray(a)      # (2,4)
    b = np.asarray(b)

    theta = q[2]
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    J = np.zeros((4, 3))

    for i in range(4):
        # Direction vector from platform anchor to fixed anchor
        Ci = q[:2] + R @ b[:, i]          # platform anchor position
        vec = a[:, i] - Ci                # vector along cable
        dist = np.linalg.norm(vec)

        if dist < 1e-9:
            u = np.zeros(2)               # singular case
        else:
            u = vec / dist                # unit vector toward fixed point

        # ∂d/∂x = -u_x
        # ∂d/∂y = -u_y
        J[i, 0] = -u[0]
        J[i, 1] = -u[1]

        # ∂d/∂θ = -u × (R b_i)   cross product in 2D = u_y * (R b)_x - u_x * (R b)_y
        Rb = R @ b[:, i]
        J[i, 2] = - (u[0] * Rb[1] - u[1] * Rb[0])   # = - (u × Rb)

    return J



def inverse_kinematics(qd, a, b):
    """
    Compute desired cable lengths.

    Parameters
    ----------
    qd : array-like, shape (3,)
        Desired platform pose [x, y, theta]
    a : array-like, shape (2, 4)
        Anchor points in inertial frame
    b : array-like, shape (2, 4)
        Anchor points in body frame

    Returns
    -------
    d : ndarray, shape (4,)
        Desired cable lengths
    """

    qd = np.asarray(qd)
    a = np.asarray(a)
    b = np.asarray(b)

    d = np.zeros(4)

    while len(qd)<3:
        qd = np.append(qd, 0.0)
    theta = qd[2]
    R_BI_d = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    for i in range(4):
        d[i] = np.linalg.norm(
            a[:, i] - qd[0:2] - R_BI_d @ b[:, i]
        )

    return d


def get_q_des(t):
    trajectory_mode = 0  # choose trajectory type

    #-- Circular trajectory ---
    if trajectory_mode == 0:
        x = p.R*np.sin(p.omega*t)
        y = p.R*np.cos(p.omega*t)-0.05
        theta = 0.2*np.sin(p.omega*t)
        q = np.array([x, y, theta])

        x_dot = p.R*p.omega*np.cos(p.omega*t)
        y_dot = -p.R*p.omega*np.sin(p.omega*t)
        theta_dot = p.r_d*p.omega*np.cos(p.omega*t)
        q_dot = np.array([x_dot, y_dot, theta_dot])
        return q, q_dot

    #-- back and forth trajectory ---
    elif trajectory_mode == 1:
        x = p.R*np.sin(p.omega*t)
        y = -0.05
        theta = p.r_d*np.sin(p.omega*t)
        q = np.array([x, y, theta])

        x_dot = p.R*p.omega*np.cos(p.omega*t)
        y_dot = 0
        theta_dot = p.r_d*p.omega*np.cos(p.omega*t)
        q_dot = np.array([x_dot, y_dot, theta_dot])
        return q, q_dot
    
    # -- rose pattern trajectory ---
    elif trajectory_mode == 2:
        f = t*np.pi*0.5*p.omega
        a = 0.2
        k = 4
        x = a*np.cos(k*f)*np.cos(f)
        y = a*np.cos(k*f)*np.sin(f)
        theta = 0.0
        q = np.array([x, y, theta])

        x_dot = -a*k*np.sin(k*f)*np.cos(f) - a*np.cos(k*f)*np.sin(f)
        y_dot = -a*k*np.sin(k*f)*np.sin(f) + a*np.cos(k*f)*np.cos(f)
        theta_dot = 0.0
        q_dot = np.array([x_dot, y_dot, theta_dot])

        return q, q_dot
    
    # -- nothing --
    elif trajectory_mode == 3:
        q = np.array([0, -0.1, 0])
        q_dot = np.array([0, 0, 0])
        return q, q_dot
    

    # -- triangle trajectory (constant speed) ---
    elif trajectory_mode == 4:

        # --- triangle definition ---
        L = 0.2  # side length [m]
        v = 0.2   # constant speed [m/s]

        # triangle vertices (equilateral, centered)
        h = np.sqrt(3)/2 * L

        p0 = np.array([-L/2, -h/3])
        p1 = np.array([ L/2, -h/3])
        p2 = np.array([ 0.0,  2*h/3])

        points = [p0, p1, p2]

        # --- timing ---
        T_edge = L / v
        T_total = 3 * T_edge

        t_mod = t % T_total

        # determine which edge
        if t_mod < T_edge:
            i = 0
            tau = t_mod
        elif t_mod < 2*T_edge:
            i = 1
            tau = t_mod - T_edge
        else:
            i = 2
            tau = t_mod - 2*T_edge

        # current segment
        p_start = points[i]
        p_end   = points[(i+1) % 3]

        direction = p_end - p_start
        direction = direction / np.linalg.norm(direction)

        # position
        pos = p_start + direction * v * tau

        # velocity
        vel = direction * v

        # small offset (like your other trajectories)
        x = pos[0]
        y = pos[1] - 0.05
        theta = 0.0

        q = np.array([x, y, theta])

        x_dot = vel[0]
        y_dot = vel[1]
        theta_dot = 0.0

        q_dot = np.array([x_dot, y_dot, theta_dot])

        return q, q_dot
    
    elif trajectory_mode == 5:
        if t % 14 < 7:
            q = np.array([0.15, -0.1, 0])
        else:
            q = np.array([-0.15, -0.1, 0])
        q_dot = np.array([0, 0, 0])
        return q, q_dot



def get_q_des_ball(t, trajectory = None, last_plan_time=0.0, q = None, q_dot = None, p_hit = None, t_hit = None, last_t_hit = None):
    # -- ball bouncing trajectory ---
    time_since_plan = (t > last_plan_time + 0.1) and (t < last_t_hit)
    past_hit_time = t >= last_t_hit+0.1
    need_new_traj = trajectory is None

    """
    if time_since_plan:
        print(f"Planning new trajectory at t={t:.2f}s, hit in {t_hit - t:.2f}s")
    if past_hit_time:
        print(f"Past hit time at t={t:.2f}s",)
    if need_new_traj:
        print(f"Need new trajectory at t={t:.2f}s",)"""

    if time_since_plan or past_hit_time or need_new_traj:
        hit_speed, theta = traj.compute_hit_velocity(p_hit)
        p_hit = np.array([p_hit[0], p_hit[1], theta])
        hit_speed = np.array([hit_speed[0], hit_speed[1], 0.0])

        direct_traj = traj.HermiteTrajectory(q, q_dot, p_hit, hit_speed, t_hit-t, t)
        if traj.is_trajectory_inside_workspace(direct_traj):
            trajectory = direct_traj
        
        else:
            final_move_duration = 0.2

            displacement = 0.5 * hit_speed * final_move_duration

            wait_point = p_hit - displacement

            # Quick move to wait point
            move_duration = min(0.4, (t_hit - t) * 0.3)
            to_wait_traj = traj.HermiteTrajectory(
                q, q_dot, wait_point, np.zeros(3), move_duration, t
            )

            remain_duration = (t_hit - t) - move_duration
            if remain_duration < final_move_duration + 0.05:
                trajectory = direct_traj
            else:
                wait_dur = remain_duration - final_move_duration

                # Final move
                final_start_t = t + move_duration + wait_dur
                from_wait_traj = traj.HermiteTrajectory(
                    wait_point, np.zeros(3), p_hit, hit_speed, final_move_duration, final_start_t
                )

                # Check if final path stays in workspace (optional but recommended)
                if not traj.is_trajectory_inside_workspace(from_wait_traj):
                    print("Warning: final ramp violates workspace – adjust duration or wait point")
                    # Fallback or shorten duration slightly

                trajectory = {
                    'type': 'chained',
                    'to_wait': to_wait_traj,
                    'wait_point': wait_point,
                    'wait_start': t + move_duration,
                    'from_wait_start': final_start_t,
                    'from_wait': from_wait_traj
                }
        last_t_hit = t_hit
        last_plan_time = t
    
    # Evaluate
    q = traj.evaluate_trajectory(trajectory, t)
    
    return q, trajectory, last_plan_time, last_t_hit


def input_q_des(q_des, input_mode, screen_center = None):
    step = 0.005

    # ---- keyboard control ----
    if input_mode == 0:
        move = np.zeros(3)

        if keyboard.is_pressed('w'):
            move[1] += step
        if keyboard.is_pressed('s'):
            move[1] -= step
        if keyboard.is_pressed('a'):
            move[0] -= step
        if keyboard.is_pressed('d'):
            move[0] += step
        if keyboard.is_pressed('q'):
            move[2] += step
        if keyboard.is_pressed('e'):
            move[2] -= step

        q_des += move
    
    # ---- Mouse control ----
    if input_mode == 1:
        scale = 0.00005

        mx, my = mouse.get_position()

        q_des[0] += (mx - screen_center[0]) * scale
        q_des[1] -= (my - screen_center[1]) * scale

        mouse.move(screen_center[0], screen_center[1])

    q_des[0] = np.clip(q_des[0], p.WORKSPACE_X_MIN, p.WORKSPACE_X_MAX)
    q_des[1] = np.clip(q_des[1], p.WORKSPACE_Y_MIN, p.WORKSPACE_Y_MAX)

    return q_des



def phi_from_d(d_des, phi0):
    phi = []
    for i in range(4):
        phi.append(
            p.motor_signs[i] * d_des[i] / (2*np.pi*p.r_d) + phi0[i]
        )
    return phi

def d_from_phi(phi, phi0):
    d = []
    for i in range(4):
        d.append(
            (phi[i]-phi0[i]) * (2*np.pi*p.r_d) * p.motor_signs[i]
        )
    return d


def direct_kinematics(l_act, v_cable, a, b, q_prev):
    """
    Direct kinematics + velocity estimation for 4-cable CDPR

    Parameters
    ----------
    l_act : array-like (4,)
        Measured cable lengths
    v_cable : array-like (4,)
        Measured cable length rates
    a : array-like (2,4)
        Frame anchor points
    b : array-like (2,4)
        Platform anchor points
    q_prev : array-like (3,)
        Previous pose estimate [x, y, theta]

    Returns
    -------
    q_est : ndarray (3,)
        Estimated pose
    qdot_est : ndarray (3,)
        Estimated pose velocity
    """

    # --- KEEP anchors as (2,4) ---
    a = np.asarray(a)
    b = np.asarray(b)
    l_act = np.asarray(l_act)
    v_cable = np.asarray(v_cable)

    m = 4
    q = np.array(q_prev, dtype=float)

    lambda_lm = 1e-3
    eps_fd = 1e-6
    tol = 1e-6

    # --- Levenberg–Marquardt pose estimation ---
    for _ in range(6):

        r = np.zeros(m)
        R = utils.rot(q[2])
        rpos = q[:2]

        for i in range(m):
            Ci = rpos + R @ b[:, i]
            r[i] = np.linalg.norm(a[:, i] - Ci) - l_act[i]

        if 0.5 * (r @ r) < tol:
            break

        # Jacobian (finite difference)
        J = np.zeros((m, 3))
        for k in range(3):
            dq = np.zeros(3)
            dq[k] = eps_fd
            q2 = q + dq

            R2 = utils.rot(q2[2])
            rpos2 = q2[:2]

            r2 = np.zeros(m)
            for i in range(m):
                Ci2 = rpos2 + R2 @ b[:, i]
                r2[i] = np.linalg.norm(a[:, i] - Ci2) - l_act[i]

            J[:, k] = (r2 - r) / eps_fd

        H = J.T @ J
        g = J.T @ r

        delta = -np.linalg.solve(H + lambda_lm * np.eye(3), g)
        q_new = q + delta

        # Check improvement
        Rn = utils.rot(q_new[2])
        r_new = np.zeros(m)
        for i in range(m):
            Ci2 = q_new[:2] + Rn @ b[:, i]
            r_new[i] = np.linalg.norm(a[:, i] - Ci2) - l_act[i]

        if (r_new @ r_new) < (r @ r):
            q = q_new
            lambda_lm /= 10
        else:
            lambda_lm *= 10

        if np.linalg.norm(delta) < 1e-7:
            break

    q_est = q.copy()

    # --- Velocity estimation ---
    R = utils.rot(q[2])
    rpos = q[:2]

    A = np.zeros((m, 3))
    for i in range(m):
        Ci = rpos + R @ b[:, i]
        u = a[:, i] - Ci
        un = np.linalg.norm(u)
        if un > 1e-9:
            u /= un
        else:
            u[:] = 0.0

        b_i = R @ b[:, i]
        tau = b_i[0] * u[1] - b_i[1] * u[0]

        A[i, :] = [u[0], u[1], tau]

    qdot_est = -np.linalg.pinv(A) @ v_cable

    return q_est, qdot_est

