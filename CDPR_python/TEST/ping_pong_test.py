import motor_actions as motor
import camera as cam
import parameters as p
import geometry as geom
import trajectory_planner as traj
import utils
import cv2
import control_loop as ctrl

import numpy as np
import threading
import time
import queue
from odrive.utils import dump_errors
from odrive.enums import *


class SharedHit:
    def __init__(self):
        self.lock = threading.Lock()
        self.p_hit = None
        self.t_hit = None
        self.center = None
        self.desired_pos_px = None  # desired position in pixel coordinates

    def set(self, p_hit, t_hit, center, desired_pos_px=None):
        with self.lock:
            self.p_hit = np.array(p_hit, dtype=float)
            self.t_hit = 0 if (t_hit == None) else float(t_hit)
            self.center = np.array(center, dtype=float)
            self.desired_pos_px = desired_pos_px

    def get(self):
        with self.lock:
            return self.p_hit, self.t_hit, self.center, self.desired_pos_px
        



def camera_tracking_loop(tracker, shared_hit, stop_event, frame_queue=None, bounce=True):
    print("Camera thread started")

    print("Camera thread started")
    cap = cam.CameraCapture(device=2)
    time.sleep(0.5)
    try:
        while not stop_event.is_set():
            start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                continue

            ret, frame = cap.read()
            if not ret:
                continue

            if bounce:
                p_hit, t_hit, center, pred_px, traj_px = tracker.update_bounce(frame)
                if center is not None:
                    shared_hit.set(p_hit, t_hit, center)
            else:
                center, pred_px = tracker.update_balance(frame)
                if center is not None:
                    shared_hit.set(0, 0, center)

            # ---- Get desired position from control loop if available ----
            _, _, _, desired_pos_px = shared_hit.get()

            # ---- annotate frame ----
            tracker.draw_overlay(frame, center, pred_px, desired_pos_px, traj_px)

            # hand frame back for display
            if frame_queue is not None:
                try:
                    frame_queue.put_nowait(frame)
                except queue.Full:
                    pass
            time.sleep(0)
            fps = 1/(time.perf_counter()-start)
            print(fps)
    finally:
        cap.release()
        cv2.destroyAllWindows()



def camera_tracking_loop(
        tracker,
        stop_event,
        frame_queue=None):

    print("Camera thread started")

    cap = cam.CameraCapture(device=2)

    time.sleep(0.5)

    prev_t = time.perf_counter()

    try:

        while not stop_event.is_set():

            # ========================================================
            # TIMING
            # ========================================================

            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now

            # ========================================================
            # FRAME ACQUISITION
            # ========================================================

            ret, frame = cap.read()

            if not ret:
                continue

            # ========================================================
            # TRACKER UPDATE
            # ========================================================

            tracker.update(frame, dt)

            # ========================================================
            # OVERLAYS
            # ========================================================

            tracker.draw_overlay(frame)

            # ========================================================
            # DISPLAY FRAME
            # ========================================================

            if frame_queue is not None:

                try:
                    frame_queue.put_nowait(frame)

                except queue.Full:
                    pass

            time.sleep(0)

    finally:

        cap.release()
        cv2.destroyAllWindows()



def run_position_control_loop_no_log(odrvs, motors, phi0, shared_hit, t0, tracker, frame_queue=None):
    print("Starting control loop")
    print("Press Ctrl+C to stop")


    d_0 = geom.inverse_kinematics([0,0,0],p.a,p.b)
    d_des = d_0.copy()
    q_prev = np.array([0, -0.2, 0])
    q_dot_des = np.array([0,0,0])
    theta = 0

    target_dt = 0.02
    dt = target_dt

    # ---- tuning window (safe: created & read from main thread) ----
    cv2.namedWindow("tuning", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("tuning", 600, 400)
    cv2.createTrackbar("Kp", "tuning", 50, 1000, lambda x: None)  # 0.0–10.0
    cv2.createTrackbar("Kd", "tuning", 30, 1000, lambda x: None)    # 0.0–10.0
    cv2.createTrackbar("Kx", "tuning", 15, 200, lambda x: None)    # 0.0–20.0
    cv2.createTrackbar("Kv", "tuning", 20, 200, lambda x: None)    # 0.0–20.0
    cv2.createTrackbar("K_v", "tuning", 10, 100, lambda x: None)   # 0.0–10.0
    cv2.createTrackbar("alpha_pos", "tuning", 40, 100, lambda x: None)  # 0.0–1.0
    cv2.createTrackbar("alpha_vel", "tuning", 60, 100, lambda x: None)  # 0.0–1.0

    # Create blank image for displaying parameter values
    tuning_display = np.zeros((100, 600, 3), dtype=np.uint8)

    # default fallbacks
    Kp = 1.5
    Kd = 0.4
    Kx = 1.5
    Kv = 2.0
    K_v = 0.5
    alpha_pos = 0.6
    alpha_vel = 0.6

    x_des = 0

    max_qdot = np.array([0.5, 0.5, 2])

    vel_history = []
    vel_window = 3


    try:
        motor.position_control(motors)
        time.sleep(0.1)

        # display camera frames if provided
        if frame_queue is not None and not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imshow("Robot vision", frame)
        cv2.waitKey(1)

        t = time.perf_counter() - t0 + 3
        ball_pos, ball_speed = geom.get_q_des(t)
        ball_pos = np.asarray(ball_pos)
        x_ref = ball_pos[0]
        y_ref = ball_pos[1]-0.1

        pos = np.array([0.0, y_ref])
        prev_pos = np.array([0.0, y_ref])
        vel_prev = np.array([0.0, 0.0])

        ctrl.smooth_move_to_pose(motors, phi0, np.array([x_ref, y_ref, 0]))

        next_t = time.perf_counter()
        loop_start = time.perf_counter()
        
        while True:

            t = time.perf_counter()-t0
            loop_start = time.perf_counter()

            
            # ---- read tuning parameters from trackbars ----
            Kp = cv2.getTrackbarPos("Kp", "tuning") / 100.0
            Kd = cv2.getTrackbarPos("Kd", "tuning") / 100.0
            Kx = cv2.getTrackbarPos("Kx", "tuning") / 10.0
            Kv = cv2.getTrackbarPos("Kv", "tuning") / 10.0
            K_v = cv2.getTrackbarPos("K_v", "tuning") / 10.0
            alpha_pos = cv2.getTrackbarPos("alpha_pos", "tuning") / 100.0
            alpha_vel = cv2.getTrackbarPos("alpha_vel", "tuning") / 100.0
            
            # Display parameter values on tuning window
            tuning_display[:] = 0  # Clear image
            cv2.putText(tuning_display, f"Kp={Kp:.2f}  Kd={Kd:.2f}  Kx={Kx:.2f}  Kv={Kv:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(tuning_display, f"K_v={K_v:.2f}  alpha_pos={alpha_pos:.2f}  alpha_vel={alpha_vel:.2f}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("tuning", tuning_display)
            
            # show camera frame if available
            if not frame_queue.empty():
                frame = frame_queue.get()
                cv2.imshow("Robot vision", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0)

            for i, axis in motors.items():
                if axis.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                    dump_errors(odrvs[i])
                    raise Exception(f"Motor {i} not in CLOSED LOOP CONTROL")


            p_hit_shared, t_hit_shared, center_shared, _ = shared_hit.get()

            vel = np.array([0, 0])

            if center_shared is not None:
                pos_raw = tracker.pixel_to_plane(center_shared)
            else:
                pos_raw = pos


            pos_pred = pos_raw + vel_prev * p.cam_latency

            pos = alpha_pos * pos_pred + (1-alpha_pos) * prev_pos
            vel_raw = (pos - prev_pos)/dt
            if np.linalg.norm(vel_raw) > p.MAX_VEL:
                vel_raw = vel_prev
            vel = alpha_vel * vel_raw + (1-alpha_vel) * vel_prev
            
            # ---- Moving-average velocity filter: reduce high-freq noise ----
            vel_history.append(vel.copy())
            if len(vel_history) > vel_window:
                vel_history.pop(0)
            vel = np.mean(vel_history, axis=0)  # smooth velocity with moving average
            
            prev_pos = pos
            vel_prev = vel
            
            ball_pos, ball_speed = geom.get_q_des(t-p.cam_latency)
            ball_pos = np.asarray(ball_pos)
            x_ref = ball_pos[0]
            y_ref = ball_pos[1]-0.1
            v_ref = ball_speed[0]

            # Convert desired position to pixel coordinates for visualization
            desired_pos_plane = np.array([[x_ref, y_ref]], dtype=np.float32)
            desired_pos_px = cv2.perspectiveTransform(
                desired_pos_plane[None, :, :], np.linalg.inv(tracker.H)
            )[0][0]
            
            # Update shared state with desired position for display
            p_hit_shared, t_hit_shared, center_shared, _ = shared_hit.get()
            shared_hit.set(p_hit_shared, t_hit_shared, center_shared, desired_pos_px)

            ex = x_ref - pos_pred[0]
            vx = vel[0]
            ev = vx - v_ref

            Kp_eff = Kp / (1 + K_v*abs(ev))

            theta = Kp_eff*ex - Kd*ev
            theta = np.clip(theta, -0.5, 0.5)
            
            x_des = pos[0]
            #x_des += Kv * vel[0] + Kx * (pos_pred - x_des)
            #x_des += (Kx*(pos[0] - x_des) + Kv*vel[0]) * dt     # platform follows ball
            #x_des += (Kx*(x_ref - x_des) - Kv*(v_ref - vx_plat)) * dt    # platform follows trajectory
            #vx_plat = (x_des - q_prev[0])/dt
            #x_des = x_ref
            # Platform motion: integrate trajectory velocity with position feedback
            # v_ref is feedforward from the trajectory (prevents lag)
            # Kx term corrects for platform position vs ball
            # Kv term damps ball velocity
            #x_des += (v_ref + Kx*(pos[0] - x_ref) + Kv*vel[0]) * dt
            x_des = np.clip(x_des, p.WORKSPACE_X_MIN, p.WORKSPACE_X_MAX)

            q_des = np.array([x_des, y_ref, -theta])

            dq = q_des - q_prev
            dq = np.clip(dq, -max_qdot*dt, max_qdot*dt)
            q_des = q_prev + dq
            q_dot_des = dq/dt

            q_prev = q_des
            d_abs = geom.inverse_kinematics(q_des,p.a,p.b)

            for i in range(4):
                d_des[i] = d_0[i] - d_abs[i]
                d_des[i] = traj.voltage_regulator(odrvs[i], d_des[i])
            
            motor.velocity_feedforward(motors, q_des, q_dot_des)
            phi_des = np.array(geom.phi_from_d(d_des, phi0))

            for i, axis in motors.items():
                axis.controller.input_pos = phi_des[i]
            


            # ---- timing ----
            next_t += target_dt

            dt = time.perf_counter() - loop_start




    except KeyboardInterrupt:
        print("\nControl loop stopped by user")

    except Exception as e:
        print("\nEXCEPTION IN CONTROL LOOP:")
        print(type(e).__name__, e)

    
    finally:
        motor.hard_stop(motors)
        cv2.destroyWindow("tuning")
        cv2.waitKey(1)
        


class BallBalanceController:
    """Encapsulates all state for the ball balancing control logic."""
    
    def __init__(self, odrvs, motors, phi0, shared_hit, t0, tracker, frame_queue=None, balance=True):
        self.odrvs = odrvs
        self.motors = motors
        self.phi0 = phi0
        self.shared_hit = shared_hit
        self.t0 = t0
        self.tracker = tracker
        self.frame_queue = frame_queue
        self.balance = balance

        # Controller state
        self.d_0 = geom.inverse_kinematics([0,0,0], p.a, p.b)
        self.d_des = self.d_0.copy()
        self.q_prev = np.array([0.0, -0.2, 0.0])
        self.q_dot_des = np.array([0.0, 0.0, 0.0])
        self._safe_q = np.array([0.0, -0.0, 0.0])

        self.Kp = 1.5
        self.Kd = 0.4
        self.K_v = 0.5
        self.alpha_pos = 0.6
        self.alpha_vel = 0.6
        self.max_qdot = np.array([1, 1, 2])

        self.vel_history = []
        self.vel_window = 3
        self.x_des = 0.0

        # Initialize position/velocity estimates
        t = time.perf_counter() - t0 + 3
        if self.balance:
            ball_pos, ball_speed = geom.get_q_des(t)
        else:
            ball_pos, ball_speed = np.array([0, -0.15]), np.array([0,0])
        ball_pos = np.asarray(ball_pos)
        self.pos = np.array([0.0, ball_pos[1] - 0.1])
        self.prev_pos = self.pos.copy()
        self.vel_prev = np.array([0.0, 0.0])

    def is_converged(self, tol_pos=0.2, tol_vel=0.002):
        """Returns True when ball is stable near target."""
        t = time.perf_counter() - self.t0
        ball_pos, _ = geom.get_q_des(t)
        ball_pos = np.asarray(ball_pos)
        x_ref = ball_pos[0]
        y_ref = ball_pos[1] - 0.1

        pos_err = np.linalg.norm(self.pos - np.array([x_ref, y_ref]))
        vel_mag = np.linalg.norm(self.vel_prev)
        print("pos error: ", pos_err, "vel_error: ", vel_mag)
        return pos_err < tol_pos and vel_mag < tol_vel

    def step(self, dt):
        """
        Run one control iteration. Call this from your master loop.
        Returns q_des for use by other controllers if needed.
        """
        t = time.perf_counter() - self.t0

        # --- camera / ball tracking ---
        p_hit_shared, t_hit_shared, center_shared, _ = self.shared_hit.get()

        self._ball_lost = center_shared is None

        if center_shared is not None:
            pos_raw = self.tracker.pixel_to_plane(center_shared)
            self._frames_without_ball = 0
        else:
            self._frames_without_ball = getattr(self, '_frames_without_ball', 0) + 1
            pos_raw = self.prev_pos  # don't move estimate

        # If ball has been gone too long, stop trying to control
        LOST_THRESHOLD = 10  # frames
        if self._frames_without_ball > LOST_THRESHOLD:
            # Hold last known safe position, zero velocity commands
            motor.motor_input(self.motors, self._safe_q, np.zeros(3), self.phi0)
            print("ball lost")
            return self._safe_q

        pos_pred = pos_raw + self.vel_prev * p.cam_latency
        self.pos = self.alpha_pos * pos_pred + (1 - self.alpha_pos) * self.prev_pos
        vel_raw = (self.pos - self.prev_pos) / dt
        if np.linalg.norm(vel_raw) > p.MAX_VEL:
            vel_raw = self.vel_prev
        vel = self.alpha_vel * vel_raw + (1 - self.alpha_vel) * self.vel_prev

        self.vel_history.append(vel.copy())
        if len(self.vel_history) > self.vel_window:
            self.vel_history.pop(0)
        vel = np.mean(self.vel_history, axis=0)
        self.vel_prev = vel
        self.prev_pos = self.pos

        # --- trajectory reference ---
        if self.balance:
            ball_pos, ball_speed = geom.get_q_des(t)
        else:
            ball_pos, ball_speed = [0, -0.15], [0, 0]
        ball_pos = np.asarray(ball_pos)
        x_ref, y_ref = ball_pos[0], ball_pos[1] - 0.05
        v_ref = ball_speed[0]

        # --- control law ---
        ex = x_ref - pos_pred[0]
        ev = vel[0] - v_ref
        Kp_eff = self.Kp / (1 + self.K_v * abs(ev))
        theta = np.clip(Kp_eff * ex - self.Kd * ev, -0.5, 0.5)

        self.x_des = np.clip(self.pos[0], p.WORKSPACE_X_MIN, p.WORKSPACE_X_MAX)
        q_des = np.array([self.x_des, y_ref, -theta])

        dq = np.clip(q_des - self.q_prev, -self.max_qdot * dt, self.max_qdot * dt)
        q_des = self.q_prev + dq
        q_dot_des = dq / dt
        self.q_prev = q_des

        # --- IK + motor commands ---
        d_abs = geom.inverse_kinematics(q_des, p.a, p.b)
        for i in range(4):
            self.d_des[i] = self.d_0[i] - d_abs[i]
            self.d_des[i] = traj.voltage_regulator(self.odrvs[i], self.d_des[i])

        motor.velocity_feedforward(self.motors, q_des, q_dot_des)
        phi_des = np.array(geom.phi_from_d(self.d_des, self.phi0))
        for i, axis in self.motors.items():
            axis.controller.input_pos = phi_des[i]

        return q_des


class ThrowController:
    PHASE_THROW = "throw"
    PHASE_DONE  = "done"

    def __init__(self, motors, phi0, q_start, duration=0.23, throw_dist=0.18):
        self.motors     = motors
        self.phi0       = phi0
        self.q_start    = q_start.copy()
        self.duration   = duration
        self.throw_dist = throw_dist
        self.phase      = self.PHASE_THROW
        self.t_start    = None

        self._q0 = q_start.copy()
        self._qf = np.array([q_start[0], q_start[1] + throw_dist, q_start[2]])
        self._v0 = np.zeros(3)

        self._a_throw = 2.0 * throw_dist / duration**2

        print(f"[Throw] a_throw={self._a_throw:.2f} m/s²  "
              f"v_release={self._a_throw * duration:.2f} m/s")

    def start(self):
        self.t_start = time.perf_counter()
        self.phase   = self.PHASE_THROW
        print(f"[Throw] Started  q0={self._q0}  qf={self._qf}")

    def is_done(self):
        return self.phase == self.PHASE_DONE

    def step(self):
        if self.t_start is None or self.phase == self.PHASE_DONE:
            return self._qf.copy()  # hold at release point until next controller takes over

        t_rel = time.perf_counter() - self.t_start

        if t_rel <= self.duration-0.05:
            self.phase = self.PHASE_THROW
            y    = self._q0[1] + 0.5 * self._a_throw * t_rel**2
            ydot = self._a_throw * t_rel
        elif t_rel <= self.duration:
            y    = self._q0[1] + 0.5 * self._a_throw * self.duration**2
            ydot = 0
        else:
            self.phase = self.PHASE_DONE
            print("[Throw] Complete")
            return self._qf.copy()

        q_des     = np.array([self._q0[0], y,    self._q0[2]])
        q_dot_des = np.array([0.0,         ydot, 0.0        ])
        motor.motor_input(self.motors, q_des, q_dot_des, self.phi0)
        return q_des




class BallBounceController:

    def __init__(self, odrvs, motors, phi0, shared_hit, t0, tracker,
                 q_start=None, hit_vel=0.8,
                 ready_y_offset=-0.18):

        # --- hardware ---
        self.odrvs = odrvs
        self.motors = motors
        self.phi0 = phi0

        # --- state ---
        self.q_des = q_start.copy() if q_start is not None else np.array([0.0, -0.1, 0.0])
        self.q_dot_des = np.zeros(3)

        # --- targets  ---
        self.q_target = self.q_des.copy()
        self.q_dot_target = np.zeros(3)

        # --- motion parameters ---
        self.Kp = np.array([400.0, 800.0, 2000.0])
        self.Kd = np.array([40.0, 50.0, 80.0])

        self.MAX_QDOT  = np.array([3.0, 3.0, 6.0])
        self.MAX_QDDOT = np.array([50.0, 100.0, 200.0])

        # --- bounce logic ---
        self.shared_hit = shared_hit
        self.t0 = t0
        self.tracker = tracker

        self.hit_vel = hit_vel
        self.ready_y_offset = ready_y_offset

        self._p_hit = [0, -0.05]
        self._t_hit = None
        self._last_t_hit = -np.inf
        self.punch_dist = 0.2

        # --- ball state estimation ---
        self.prev_pos = np.zeros(2)
        self.pos = np.zeros(2)

        self.vel_prev = np.zeros(2)

        # smoothing
        self.vel_history = []
        self.vel_window = 3

        # filtering
        self.alpha_pos = 0.4
        self.alpha_vel = 0.5

        # lost tracking
        self._frames_without_ball = 0

        # --- theta controller ---
        self.Kp_theta = 0.5
        self.Kd_theta = 0.3
        self.K_v = 0.5
        self.theta_max = 0.5

        # --- Ball control ---
        self.Kp_theta = 0.5
        self.Kd_theta = 0.1
        self.K_v = 0.5

        self._punching = False
        self._returning = True
        self._punch_start = None
        self._punch_y0 = None

        # --- kinematics ---
        self.d_0 = geom.inverse_kinematics([0, 0, 0], p.a, p.b)
        self.d_des = self.d_0.copy()

    # ============================================================
    # MOTION FILTER
    # ============================================================
    def _apply_motion_filter(self, dt):
        """
        Smoothly moves q_des toward q_target with velocity/acceleration limits.
        """

        e = self.q_target - self.q_des

        # PD-like acceleration toward target
        q_ddot = self.Kp * e - self.Kd * self.q_dot_des

        # limit acceleration
        q_ddot = np.clip(q_ddot, -self.MAX_QDDOT, self.MAX_QDDOT)

        # integrate velocity
        self.q_dot_des += q_ddot * dt
        self.q_dot_des = np.clip(self.q_dot_des, -self.MAX_QDOT, self.MAX_QDOT)

        # integrate position
        self.q_des += self.q_dot_des * dt

    # ============================================================
    # LOW-LEVEL SEND
    # ============================================================
    def _send(self):
        d_abs = geom.inverse_kinematics(self.q_des, p.a, p.b)

        for i in range(4):
            self.d_des[i] = self.d_0[i] - d_abs[i]
            self.d_des[i] = traj.voltage_regulator(self.odrvs[i], self.d_des[i])

        phi_des = geom.phi_from_d(self.d_des, self.phi0)

        motor.velocity_feedforward(self.motors, self.q_des, self.q_dot_des)

        for i, axis in self.motors.items():
            axis.controller.input_pos = phi_des[i]

    # ============================================================
    # STATE LOGIC
    # ============================================================
    def _state_punch(self, t):
        t_rel = time.perf_counter() - self._punch_start

        y_target = self._punch_y0 + self.hit_vel * t_rel

        self.q_target = np.array([
            self.q_des[0],
            y_target,
            self.q_des[2]
        ])

        # stop condition
        if t_rel > self.punch_dist / self.hit_vel:
            self._punching = False
            self._returning = True

    def _state_return(self):

        # --- Y: return to ready height ---
        ready_y = self._p_hit[1] + self.ready_y_offset

        # --- X: track ball ---
        target_x = np.clip(self.pos[0], p.WORKSPACE_X_MIN, p.WORKSPACE_X_MAX)

        self.q_target = np.array([
            target_x,
            ready_y,
            self.q_target[2]
        ])

        # --- exit condition ---
        if abs(self.q_des[1] - ready_y) < 0.01:
            self._returning = False
            print("[Bounce] Ready")

    def _state_tracking(self, t):
        if self._p_hit is None or self._t_hit is None:
            return

        time_to_hit = self._t_hit - t

        if time_to_hit > 0.2:
            target_x = np.clip(self._p_hit[0], p.WORKSPACE_X_MIN, p.WORKSPACE_X_MAX)
            target_y = np.clip(self._p_hit[1] + self.ready_y_offset, p.WORKSPACE_Y_MIN, p.WORKSPACE_Y_MAX)

            self.q_target = np.array([target_x, target_y, self.q_des[2]])

        else:
            if not self._punching and not self._returning:
                print("[Bounce] Punching!")
                self._punching = True
                self._punch_start = time.perf_counter()
                self._punch_y0 = self.q_des[1]

    # ============================================================
    # MAIN LOOP
    # ============================================================
    def step(self, dt):

        t = time.perf_counter() - self.t0

        # --- motor safety ---
        for i, axis in self.motors.items():
            if axis.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                dump_errors(self.odrvs[i])
                raise Exception(f"Motor {i} not in CLOSED LOOP CONTROL")

        # --- update hit prediction ---
        p_hit_shared, t_hit_shared, center_shared, _ = self.shared_hit.get()
        if (p_hit_shared is not None
                and not np.any(np.isnan(np.asarray(p_hit_shared)))
                and t_hit_shared is not None
                and t_hit_shared > t
                and t_hit_shared != self._last_t_hit):
            self._p_hit = np.asarray(p_hit_shared)
            self._t_hit = t_hit_shared
        
        if center_shared is not None:
            pos_raw = self.tracker.pixel_to_plane(center_shared)
            self._frames_without_ball = 0
        else:
            self._frames_without_ball = getattr(self, '_frames_without_ball', 0) + 1
            pos_raw = self.prev_pos  # don't move estimate

        # If ball has been gone too long, stop trying to control
        LOST_THRESHOLD = 10  # frames
        if self._frames_without_ball > LOST_THRESHOLD:
            # Hold last known safe position, zero velocity commands
            motor.motor_input(self.motors, self._safe_q, np.zeros(3), self.phi0)
            print("ball lost")
            return self._safe_q

        # --- latency compensation ---
        pos_pred = pos_raw + self.vel_prev * p.cam_latency

        # --- filtering ---
        self.pos = self.alpha_pos * pos_pred + (1 - self.alpha_pos) * self.prev_pos

        vel_raw = (self.pos - self.prev_pos) / dt

        if np.linalg.norm(vel_raw) > p.MAX_VEL:
            vel_raw = self.vel_prev

        vel = self.alpha_vel * vel_raw + (1 - self.alpha_vel) * self.vel_prev

        # --- smoothing ---
        self.vel_history.append(vel.copy())
        if len(self.vel_history) > self.vel_window:
            self.vel_history.pop(0)

        vel = np.mean(self.vel_history, axis=0)

        # --- store ---
        self.prev_pos = self.pos
        self.vel_prev = vel

        # --- state machine ---
        if self._punching:
            self._state_punch(t)

        elif self._returning:
            self._state_return()

        else:
            self._state_tracking(t)

        # --- BALL CONTROL (theta) ---
        if self._p_hit is not None:
            ex_hit = -self._p_hit[0]
            ex_now = -self.pos[0]

            # blend (small weight on prediction)
            ex = 0.8 * ex_now + 0.2 * ex_hit
        else:
            ex = -self.pos[0]
        ev = vel[0]

        Kp_eff = self.Kp_theta / (1 + self.K_v * abs(ev))

        theta_cmd = -(Kp_eff * ex - self.Kd_theta * ev)

        # limit tilt
        theta_cmd = np.clip(theta_cmd, -0.5, 0.5)

        self.q_target[2] = theta_cmd

        # --- apply motion filter ---
        self._apply_motion_filter(dt)

        # --- send to motors ---
        self._send()

        return self.q_des




def run_master_loop(odrvs, motors, phi0, shared_hit, t0, tracker, frame_queue=None):
    """
    Master loop that sequences through controllers.
    Add more phases by extending the state machine.
    """
    target_dt = 0.02
    dt = target_dt

    motor.position_control(motors)
    time.sleep(0.1)

    # --- instantiate controllers up front ---
    balance = BallBalanceController(odrvs, motors, phi0, shared_hit, t0, tracker, frame_queue, balance=False)
    throw = None  # created when we're ready to throw
    bounce = BallBounceController(odrvs, motors, phi0, shared_hit, t0, tracker)

    # --- state machine ---
    PHASE_BALANCE   = "balance"
    PHASE_THROW     = "throw"
    PHASE_BOUNCE    = "bounce"

    phase = PHASE_BALANCE
    phase_start = time.perf_counter()
    prev_start = time.perf_counter()

    MAX_BALANCE_TIME = 10.0   # fallback: throw after 10s even if not converged
    MAX_BOUNCE_TIME = 10.0
    THROW_DURATION   = 0.1


    try:
        while True:
            loop_start = time.perf_counter()
            dt = loop_start-prev_start
            prev_start = loop_start

            # --- motor safety check ---
            for i, axis in motors.items():
                if axis.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                    dump_errors(odrvs[i])
                    raise Exception(f"Motor {i} not in CLOSED LOOP CONTROL")

            # --- camera display ---
            if frame_queue is not None and not frame_queue.empty():
                cv2.imshow("Robot vision", frame_queue.get())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            elapsed = time.perf_counter() - phase_start

            tracker.update(frame, dt)

            # ============================================================
            if phase == PHASE_BALANCE:
                q_des = balance.step(dt)

                should_throw = (elapsed > 5 and (balance.is_converged())         )
                if should_throw:
                    print(f"[{elapsed:.2f}s] Converged — switching to throw")
                    throw = ThrowController(motors, phi0, q_start=q_des)
                    throw.start()
                    phase = PHASE_THROW
                    phase_start = time.perf_counter()

            # ============================================================
            elif phase == PHASE_THROW:
                throw.step()

                if throw.is_done():
                    print("Throw complete")
                    bounce = BallBounceController(odrvs, motors, phi0, shared_hit, t0, tracker, q_start=throw._qf.copy())
                    phase = PHASE_BOUNCE
                    phase_start = time.perf_counter()

            # ============================================================
            elif phase == PHASE_BOUNCE:
                bounce.step(dt)

                if elapsed > MAX_BOUNCE_TIME:
                    phase = PHASE_BALANCE
                    phase_start = time.perf_counter()
            time.sleep(0)


    except KeyboardInterrupt:
        print("Stopped")
    except Exception as e:
        print(type(e).__name__, e)
    finally:
        motor.hard_stop(motors)
        cv2.destroyAllWindows()


def throw_loop(motors, phi0, q_start, duration=0.1):
    q0 = q_start
    qf = np.array([q_start[0], q_start[1]+0.2, q_start[2]])
    v0 = np.array([0.0, 0.0, 0.0])
    vf = np.array([0.0, 10.0, 0.0])

    t_start = time.perf_counter()

    traj_throw = traj.HermiteTrajectory(
        q0, v0,
        qf, vf,
        duration,
        t_start
    )
    traj_return = traj.HermiteTrajectory(
        qf, v0,
        q0, -vf,
        duration,
        t_start + duration
    )

    while True:
        t = time.perf_counter()

        if t > t_start + 2*duration + 0.1:
            break
        
        if t < t_start + duration:    
            q_des = traj.evaluate_trajectory(traj_throw, t)
            q_dot_des = traj.evaluate_velocity(traj_throw, t)
        else:
            q_des = traj.evaluate_trajectory(traj_return, t)
            q_dot_des = traj.evaluate_velocity(traj_return, t)

        motor.motor_input(motors, q_des, q_dot_des, phi0)


def throw_ball(motors, phi0):
    dt_des = 0.01
    dt = 0.01

    q0 = np.array([0.0, -0.2, 0.0])

    ctrl.smooth_move_to_pose(motors, phi0, q0, 2)
    time.sleep(0.5)

    throw = ThrowController(motors, phi0, q0)
    throw.start()
    while not throw.is_done():
        loop_start = time.perf_counter()

        throw.step()

        while time.perf_counter() - loop_start < dt_des:
            pass
        dt = time.perf_counter() - loop_start

    time.sleep(0.5)

    ctrl.smooth_move_to_pose(motors, phi0, q0,2)
    time.sleep(0.2)

    motor.hard_stop(motors)


def run_balance_loop(odrvs, motors, phi0, shared_hit, t0, tracker, frame_queue=None):
    target_dt = 0.02
    dt = target_dt

    motor.position_control(motors)
    time.sleep(0.1)

    # --- instantiate controller up front ---
    balance = BallBalanceController(odrvs, motors, phi0, shared_hit, t0, tracker, frame_queue)

    next_t = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()

            # --- motor safety check ---
            for i, axis in motors.items():
                if axis.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                    dump_errors(odrvs[i])
                    raise Exception(f"Motor {i} not in CLOSED LOOP CONTROL")

            # --- camera display ---
            if frame_queue is not None and not frame_queue.empty():
                cv2.imshow("Robot vision", frame_queue.get())
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            q_des = balance.step(dt)
            time.sleep(0)
            # --- timing ---
            dt = time.perf_counter() - loop_start

    except KeyboardInterrupt:
        print("Stopped")
    except Exception as e:
        print(type(e).__name__, e)
    finally:
        motor.hard_stop(motors)
        cv2.destroyAllWindows()


# ---- home platform using camera ----
def home_platform(motors):
    #for i, axis in motors.items():
     #   axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
      #  axis.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
       # axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
        #motor.set_motor_torque(i, axis, 0.2)

    cap = cv2.VideoCapture(0)

    H = cam.calibrate_plane(cap, 2)

    pose = cam.find_platform(cap, H)

    d_home = geom.inverse_kinematics(pose, p.a, p.b)

    d_ref = geom.inverse_kinematics([0,0,0], p.a, p.b)
    delta_d = d_ref - d_home
    phi0 = []
    for i, axis in motors.items():
        phi0.append(axis.pos_estimate - delta_d[i]*p.motor_signs[i]/(2*np.pi*p.r_d))
    motor.hard_stop(motors)
    return phi0




def ping_pong_bot(odrvs, motors, phi0):
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ctrl.smooth_move_to_pose(motors, phi0, [0,0,0])
    time.sleep(0.2)
    motor.hard_stop(motors)

    t0 = time.perf_counter()

    # ---- plane calibration ----
    H = cam.calibrate_plane(cap, duration=2.0)
    cap.release()
    time.sleep(0.5)

    shared_hit = SharedHit()

    tracker = cam.BallTracker(H, t0)

    ctrl.smooth_move_to_pose(motors, phi0, [0,-0.15,0])

    # ---- start camera thread ----
    stop_event = threading.Event()
    frame_queue = queue.Queue(maxsize=1)

    cam_thread = threading.Thread(
        target=camera_tracking_loop,
        args=(tracker, shared_hit, stop_event, frame_queue),
        daemon=True,
    )
    cam_thread.start()
    time.sleep(0.5)

    # ---- run platform control ----
    try:
        run_master_loop(
            odrvs,
            motors,
            phi0,
            shared_hit,
            t0,
            tracker,
            frame_queue
        )
    finally:
        stop_event.set()
        cam_thread.join()




def ball_balancing(odrvs, motors, phi0):
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ctrl.smooth_move_to_pose(motors, phi0, [0,0,0])
    time.sleep(0.2)
    motor.hard_stop(motors)

    t0 = time.perf_counter()

    # ---- plane calibration ----
    H = cam.calibrate_plane(cap, duration=2.0)

    cap.release()
    cv2.destroyWindow("Plane calibration")
    time.sleep(0.5)

    shared_hit = SharedHit()

    tracker = cam.BallTracker(H, t0)

    # ---- start camera thread ----
    stop_event = threading.Event()
    frame_queue = queue.Queue(maxsize=1)

    cam_thread = threading.Thread(
        target=camera_tracking_loop,
        args=(tracker, shared_hit, stop_event, frame_queue, False),
        daemon=True,
    )
    cam_thread.start()
    time.sleep(0.6)

    # ---- run platform control (GUI stays in main thread) ----
    try:
        run_position_control_loop_no_log(
            odrvs,
            motors,
            phi0,
            shared_hit,
            t0,
            tracker,
            frame_queue,
        )
    finally:
        stop_event.set()
        cam_thread.join()


def ball_balancing_class(odrvs, motors, phi0):
    cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 120)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ctrl.smooth_move_to_pose(motors, phi0, [0,0,0])
    time.sleep(0.2)
    motor.hard_stop(motors)

    t0 = time.perf_counter()

    # ---- plane calibration ----
    H = cam.calibrate_plane(cap, duration=2.0)

    cap.release()
    cv2.destroyWindow("Plane calibration")
    time.sleep(0.5)

    shared_hit = SharedHit()

    tracker = cam.BallTracker(H, t0)

    # ---- start camera thread ----
    stop_event = threading.Event()
    frame_queue = queue.Queue(maxsize=1)

    cam_thread = threading.Thread(
        target=camera_tracking_loop,
        args=(tracker, shared_hit, stop_event, frame_queue, False),
        daemon=True,
    )
    cam_thread.start()
    ctrl.smooth_move_to_pose(motors, phi0, np.array([0,-0.15,0]))

    # ---- run platform control (GUI stays in main thread) ----
    try:
        run_balance_loop(odrvs, motors, phi0, shared_hit, t0, tracker, frame_queue)
    finally:
        stop_event.set()
        cam_thread.join()