import motor_actions as motor
import camera as cam
import parameters as p
import geometry as geom
import cv2
import control_loop as ctrl

import numpy as np
import threading
import time
import queue
from odrive.utils import dump_errors
from odrive.enums import *


CAMERA_DEVICE = 2
CAMERA_FPS = 120
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

HOME_POSE = [0, 0, 0]
START_POSE = [0, -0.15, 0]

CALIBRATION_DURATION = 2.0
HOME_SETTLE_TIME = 0.2
CALIBRATION_SETTLE_TIME = 0.5
TRACKING_THREAD_SETTLE_TIME = 0.5


def open_calibration_camera():
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*'MJPG')
    )

    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    return cap


def initialize_tracking(motors, phi0):
    cap = open_calibration_camera()

    ctrl.smooth_move_to_pose(
        motors,
        phi0,
        HOME_POSE
    )

    time.sleep(HOME_SETTLE_TIME)

    motor.hard_stop(motors)

    t0 = time.perf_counter()

    try:
        H = cam.calibrate_plane(
            cap,
            duration=CALIBRATION_DURATION
        )

    finally:
        cap.release()

    time.sleep(CALIBRATION_SETTLE_TIME)

    tracker = cam.BallTracker(H, t0)

    ctrl.smooth_move_to_pose(
        motors,
        phi0,
        START_POSE
    )

    stop_event = threading.Event()
    frame_queue = queue.Queue(maxsize=1)

    cam_thread = threading.Thread(
        target=camera_tracking_loop,
        args=(
            tracker,
            stop_event,
            frame_queue
        ),
        daemon=True,
    )

    cam_thread.start()

    time.sleep(TRACKING_THREAD_SETTLE_TIME)

    return t0, tracker, frame_queue, stop_event, cam_thread


def check_motor_states(odrvs, motors):
    for i, axis in motors.items():

        if axis.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:

            dump_errors(odrvs[i])

            raise Exception(
                f"Motor {i} not in CLOSED LOOP CONTROL"
            )


def show_latest_frame(frame_queue):
    if (
        frame_queue is not None
        and
        not frame_queue.empty()
    ):

        cv2.imshow(
            "Robot vision",
            frame_queue.get()
        )

    return cv2.waitKey(1) & 0xFF == ord('q')


def camera_tracking_loop(
        tracker,
        stop_event,
        frame_queue=None):

    print("Camera thread started")

    cap = cam.CameraCapture(device=CAMERA_DEVICE)

    time.sleep(0.5)

    prev_t = time.perf_counter()

    try:

        while not stop_event.is_set():

            # TIMING

            now = time.perf_counter()
            dt = now - prev_t
            prev_t = now

            # FRAME ACQUISITION

            ret, frame = cap.read()

            if not ret:
                continue

            # TRACKER UPDATE

            tracker.update(frame, dt)

            # OVERLAYS

            tracker.draw_overlay(frame)

            # DISPLAY FRAME

            if frame_queue is not None:

                try:
                    frame_queue.put_nowait(frame)

                except queue.Full:
                    pass

            time.sleep(0)

    finally:

        cap.release()
        cv2.destroyAllWindows()



class BallBalanceController:
    """
    Ball balancing controller.

    Uses the tracker as the single source of truth for:
        - ball position
        - ball velocity
        - filtered state estimates

    This controller is responsible only for:
        - generating platform targets
        - balancing control
        - sending motor commands
    """

    def __init__(
            self,
            odrvs,
            motors,
            phi0,
            t0,
            tracker,
            frame_queue=None,
            balance=True):

        # HARDWARE

        self.odrvs = odrvs
        self.motors = motors
        self.phi0 = phi0

        # SHARED STATE

        self.t0 = t0
        self.tracker = tracker
        self.frame_queue = frame_queue

        # MODE

        self.balance = balance

        # PLATFORM STATE

        self.q_des = np.array([0.0, -0.2, 0.0])
        self.q_dot_des = np.zeros(3)

        self._safe_q = np.array([0.0, -0.15, 0.0])

        # CONTROL PARAMETERS

        self.Kp = 2
        self.Kd = 0.85
        self.K_bias = -0.2
        self.K_v = 0.0

        self.max_qdot = np.array([2.0, 2.0, 6.0])

        # KINEMATICS

        self.d_0 = geom.inverse_kinematics(
            [0, 0, 0],
            p.a,
            p.b
        )

        self.d_des = self.d_0.copy()

    # CONVERGENCE CHECK

    def is_converged(
            self,
            tol_pos=0.06,
            tol_vel=0.02):

        t = time.perf_counter() - self.t0

        if self.balance:
            ball_ref, _ = geom.get_q_des(t)
        else:
            ball_ref = np.array([0.0, -0.15])

        ball_ref = np.asarray(ball_ref)

        x_ref = ball_ref[0]
        y_ref = ball_ref[1] - 0.05

        pos = self.tracker.pos
        vel = self.tracker.vel

        pos_err = np.linalg.norm(
            pos - np.array([x_ref, y_ref])
        )

        vel_mag = np.linalg.norm(vel)

        print(
            f"pos error: {pos_err:.4f}   "
            f"vel: {vel_mag:.4f}"
        )

        return (
            pos_err < tol_pos
            and
            vel_mag < tol_vel
        )

    # LOW LEVEL SEND

    def _send(self):

        d_abs = geom.inverse_kinematics(
            self.q_des,
            p.a,
            p.b
        )

        for i in range(4):
            self.d_des[i] = self.d_0[i] - d_abs[i]
            self.d_des[i] = motor.voltage_regulator(
                self.odrvs[i],
                self.d_des[i]
            )

        phi_des = geom.phi_from_d(
            self.d_des,
            self.phi0
        )

        motor.velocity_feedforward(
            self.motors,
            self.q_des,
            self.q_dot_des
        )

        for i, axis in self.motors.items():
            axis.controller.input_pos = phi_des[i]

    # MAIN STEP

    def step(self, dt):

        t = time.perf_counter() - self.t0

        # MOTOR SAFETY

        check_motor_states(self.odrvs, self.motors)

        # TRACKER STATE

        if self.tracker.frames_without_ball > 10:

            motor.motor_input(
                self.motors,
                self._safe_q,
                np.zeros(3),
                self.phi0
            )

            print("ball lost")

            return self._safe_q

        pos = self.tracker.pos
        vel = self.tracker.vel

        # REFERENCE TRAJECTORY

        if self.balance:

            ball_ref, ball_vel_ref = geom.get_q_des(t)

        else:

            ball_ref = np.array([0.0, -0.15])
            ball_vel_ref = np.array([0.0, 0.0])

        ball_ref = np.asarray(ball_ref)

        x_ref = ball_ref[0]
        y_ref = ball_ref[1] - 0.044

        v_ref = ball_vel_ref[0]

        # BALL BALANCING CONTROL

        ex = x_ref - pos[0]
        #print("pos: ", pos[0], " e: ", ex)
        ev = vel[0] - v_ref

        Kp_eff = self.Kp / (1 + self.K_v * abs(ev))

        theta = np.clip(Kp_eff * ex - self.Kd * ev + self.K_bias*pos[0], -0.3, 0.3)     #TODO: add feedforward

        # PLATFORM TARGET

        x_des = np.clip(
            pos[0],
            p.WORKSPACE_X_MIN,
            p.WORKSPACE_X_MAX
        )

        q_target = np.array([
            x_des,
            y_ref,
            -theta
        ])

        # VELOCITY LIMITING

        dq = np.clip(q_target - self.q_des, -self.max_qdot * dt, self.max_qdot * dt)

        self.q_des += dq

        self.q_dot_des = dq / max(dt, 1e-4)

        # SEND COMMANDS

        self._send()

        return self.q_des, ball_ref, ball_vel_ref


class ThrowController:

    PHASE_THROW = "throw"
    PHASE_HOLD  = "hold"
    PHASE_DONE  = "done"

    def __init__(
            self,
            motors,
            phi0,
            q_start,
            duration=0.23,
            throw_dist=0.18,
            hold_time=0.05):

        # HARDWARE

        self.motors = motors
        self.phi0 = phi0

        # THROW PARAMETERS

        self.duration = duration
        self.throw_dist = throw_dist
        self.hold_time = hold_time

        # TRAJECTORY

        self.q_start = q_start.copy()

        self.q_release = np.array([
            q_start[0],
            q_start[1] + throw_dist,
            q_start[2]
        ])

        # constant acceleration profile
        self.a_throw = (
            2.0 * throw_dist
            / duration**2
        )

        self.v_release = (
            self.a_throw * duration
        )

        # STATE

        self.phase = self.PHASE_THROW

        self.t_start = None
        self.t_release = None

        print(
            f"[Throw] a_throw = {self.a_throw:.2f} m/s²   "
            f"v_release = {self.v_release:.2f} m/s"
        )

    # START

    def start(self):

        self.t_start = time.perf_counter()

        self.phase = self.PHASE_THROW

        print(
            f"[Throw] Started   "
            f"q0 = {self.q_start}   "
            f"qf = {self.q_release}"
        )

    # STATUS

    def is_done(self):

        return self.phase == self.PHASE_DONE

    # MAIN STEP

    def step(self):

        # IDLE

        if self.t_start is None:

            return self.q_start.copy()

        # FINISHED

        if self.phase == self.PHASE_DONE:

            return self.q_release.copy()

        # TIME

        t_rel = (
            time.perf_counter()
            - self.t_start
        )

        # THROW PHASE

        if self.phase == self.PHASE_THROW:

            if t_rel <= self.duration:

                y = (
                    self.q_start[1]
                    +
                    0.5 * self.a_throw * t_rel**2
                )

                ydot = (
                    self.a_throw * t_rel
                )

            else:

                # clamp exactly at release
                y = self.q_release[1]
                ydot = 0.0

                self.phase = self.PHASE_HOLD

                self.t_release = time.perf_counter()

                print("[Throw] Release")

        # HOLD PHASE

        if self.phase == self.PHASE_HOLD:

            y = self.q_release[1]
            ydot = 0.0

            if (
                time.perf_counter()
                - self.t_release
                > self.hold_time
            ):

                self.phase = self.PHASE_DONE

                print("[Throw] Complete")

        # COMMAND

        q_des = np.array([
            self.q_start[0],
            y,
            self.q_start[2]
        ])

        q_dot_des = np.array([
            0.0,
            ydot,
            0.0
        ])

        # SEND

        motor.motor_input(
            self.motors,
            q_des,
            q_dot_des,
            self.phi0
        )

        return q_des



class BallBounceController:
    """
    Ball bouncing controller.

    Uses tracker as the centralized world-state estimator.

    Tracker provides:
        - ball position
        - ball velocity
        - predicted hit point
        - predicted hit timing

    This controller handles:
        - bounce state machine
        - platform trajectory generation
        - theta stabilization
        - motion filtering
        - motor commands
    """

    def __init__(
            self,
            odrvs,
            motors,
            phi0,
            t0,
            tracker,
            q_start=None,
            hit_vel=0.8,
            ready_y_offset=-0.18):

        # HARDWARE

        self.odrvs = odrvs
        self.motors = motors
        self.phi0 = phi0

        # SHARED STATE

        self.t0 = t0
        self.tracker = tracker

        # PLATFORM STATE

        self.q_des = (
            q_start.copy()
            if q_start is not None
            else np.array([0.0, -0.1, 0.0])
        )

        self.q_dot_des = np.zeros(3)

        self.q_target = self.q_des.copy()

        self._safe_q = self.q_des.copy()

        # MOTION FILTER PARAMETERS

        self.Kp = np.array([
            400.0,
            800.0,
            2000.0
        ])

        self.Kd = np.array([
            40.0,
            50.0,
            80.0
        ])

        self.MAX_QDOT = np.array([
            3.0,
            3.0,
            6.0
        ])

        self.MAX_QDDOT = np.array([
            50.0,
            100.0,
            200.0
        ])

        # THETA CONTROL

        self.Kp_theta = 0.5
        self.Kd_theta = 0.1
        self.K_v = 0.5

        self.theta_max = 0.5

        # BOUNCE PARAMETERS

        self.hit_vel = hit_vel
        self.ready_y_offset = ready_y_offset

        self.punch_dist = 0.2

        # STATE MACHINE

        self._punching = False
        self._returning = True

        self._punch_start = None
        self._punch_y0 = None

        # KINEMATICS

        self.d_0 = geom.inverse_kinematics(
            [0, 0, 0],
            p.a,
            p.b
        )

        self.d_des = self.d_0.copy()

    # MOTION FILTER

    def _apply_motion_filter(self, dt):

        e = self.q_target - self.q_des

        q_ddot = (
            self.Kp * e
            -
            self.Kd * self.q_dot_des
        )

        q_ddot = np.clip(
            q_ddot,
            -self.MAX_QDDOT,
            self.MAX_QDDOT
        )

        self.q_dot_des += q_ddot * dt

        self.q_dot_des = np.clip(
            self.q_dot_des,
            -self.MAX_QDOT,
            self.MAX_QDOT
        )

        self.q_des += self.q_dot_des * dt

    # LOW LEVEL SEND

    def _send(self):

        d_abs = geom.inverse_kinematics(
            self.q_des,
            p.a,
            p.b
        )

        for i in range(4):

            self.d_des[i] = (
                self.d_0[i] - d_abs[i]
            )

            self.d_des[i] = motor.voltage_regulator(
                self.odrvs[i],
                self.d_des[i]
            )

        phi_des = geom.phi_from_d(
            self.d_des,
            self.phi0
        )

        motor.velocity_feedforward(
            self.motors,
            self.q_des,
            self.q_dot_des
        )

        for i, axis in self.motors.items():
            axis.controller.input_pos = phi_des[i]

    # PUNCH STATE

    def _state_punch(self):

        t_rel = (
            time.perf_counter()
            - self._punch_start
        )

        y_target = (
            self._punch_y0
            +
            self.hit_vel * t_rel
        )

        self.q_target = np.array([
            self.q_des[0],
            y_target,
            self.q_target[2]
        ])

        if t_rel > self.punch_dist / self.hit_vel:

            self._punching = False
            self._returning = True

    # RETURN STATE

    def _state_return(self):

        p_hit = self.tracker.p_hit

        if p_hit is None:
            return

        ready_y = (
            p_hit[1]
            +
            self.ready_y_offset
        )

        target_x = np.clip(
            self.tracker.pos[0],
            p.WORKSPACE_X_MIN,
            p.WORKSPACE_X_MAX
        )

        self.q_target = np.array([
            target_x,
            ready_y,
            self.q_target[2]
        ])

        if abs(self.q_des[1] - ready_y) < 0.01:

            self._returning = False

            print("[Bounce] Ready")

    # TRACKING STATE

    def _state_tracking(self, t):

        p_hit = self.tracker.p_hit
        t_hit = self.tracker.t_hit

        if p_hit is None or t_hit is None:
            return

        time_to_hit = t_hit - t

        # MOVE TO READY POSITION

        if time_to_hit > 0.2:

            target_x = np.clip(
                p_hit[0],
                p.WORKSPACE_X_MIN,
                p.WORKSPACE_X_MAX
            )

            target_y = np.clip(
                p_hit[1] + self.ready_y_offset,
                p.WORKSPACE_Y_MIN,
                p.WORKSPACE_Y_MAX
            )

            self.q_target = np.array([
                target_x,
                target_y,
                self.q_target[2]
            ])

        # START PUNCH

        else:

            if (
                not self._punching
                and
                not self._returning
            ):

                print("[Bounce] Punching!")

                self._punching = True

                self._punch_start = time.perf_counter()

                self._punch_y0 = self.q_des[1]

    # MAIN STEP

    def step(self, dt):

        t = time.perf_counter() - self.t0

        # MOTOR SAFETY

        check_motor_states(self.odrvs, self.motors)

        # BALL LOST

        if self.tracker.frames_without_ball > 10:

            motor.motor_input(
                self.motors,
                self._safe_q,
                np.zeros(3),
                self.phi0
            )

            print("ball lost")

            return self._safe_q

        # TRACKER STATE

        pos = self.tracker.pos
        vel = self.tracker.vel

        p_hit = self.tracker.p_hit

        # STATE MACHINE

        if self._punching:

            self._state_punch()

        elif self._returning:

            self._state_return()

        else:

            self._state_tracking(t)

        # THETA CONTROL

        if p_hit is not None:

            ex_hit = -p_hit[0]
            ex_now = -pos[0]

            ex = (
                0.8 * ex_now
                +
                0.2 * ex_hit
            )

        else:

            ex = -pos[0]

        ev = vel[0]

        Kp_eff = self.Kp_theta / (
            1 + self.K_v * abs(ev)
        )

        theta_cmd = -(
            Kp_eff * ex
            -
            self.Kd_theta * ev
        )

        theta_cmd = np.clip(
            theta_cmd,
            -self.theta_max,
            self.theta_max
        )

        self.q_target[2] = theta_cmd

        # MOTION FILTER

        self._apply_motion_filter(dt)

        # SEND COMMANDS

        self._send()

        return self.q_des


def run_master_loop(
        odrvs,
        motors,
        phi0,
        t0,
        tracker,
        frame_queue=None):
    """
    Master controller loop.

    Sequences between:
        - balance
        - throw
        - bounce

    The tracker thread continuously updates:
        - ball position
        - velocity
        - bounce prediction

    Controllers consume tracker state directly.
    """

    # LOOP TIMING + LOGGING

    dt = 0.02

    t_log = []

    pos_log = []
    pos_raw_log = []
    vel_log = []
    vel_raw_log = []

    p_hit_log = []
    q_log = []

    # MOTOR MODE

    motor.position_control(motors)

    time.sleep(0.1)

    # CONTROLLERS

    balance = BallBalanceController(
        odrvs,
        motors,
        phi0,
        t0,
        tracker,
        frame_queue,
        balance=False
    )

    throw = None

    bounce = BallBounceController(
        odrvs,
        motors,
        phi0,
        t0,
        tracker
    )

    # STATES

    PHASE_BALANCE = "balance"
    PHASE_THROW   = "throw"
    PHASE_BOUNCE  = "bounce"

    phase = PHASE_BALANCE

    # TIMERS

    phase_start = time.perf_counter()
    prev_start = time.perf_counter()


    MAX_BOUNCE_TIME = 65

    # MAIN LOOP

    try:

        while True:

            # LOOP TIMING

            loop_start = time.perf_counter()

            dt = loop_start - prev_start

            prev_start = loop_start

            # MOTOR SAFETY

            check_motor_states(odrvs, motors)

            # DISPLAY CAMERA FRAME

            if show_latest_frame(frame_queue):
                break

            # PHASE TIME

            elapsed = (
                time.perf_counter()
                - phase_start
            )

            # BALANCE

            if phase == PHASE_BALANCE:

                q_des, _, _ = balance.step(dt)

                should_throw = (
                    elapsed > 5.0
                    and
                    balance.is_converged()
                )

                if should_throw:

                    print(
                        f"[{elapsed:.2f}s] "
                        f"Converged — switching to throw"
                    )

                    throw = ThrowController(
                        motors,
                        phi0,
                        q_start=q_des
                    )

                    throw.start()

                    phase = PHASE_THROW

                    phase_start = time.perf_counter()

            # THROW

            elif phase == PHASE_THROW:

                q_des = throw.step()

                if throw.is_done():

                    print("Throw complete")

                    bounce = BallBounceController(
                        odrvs,
                        motors,
                        phi0,
                        t0,
                        tracker,
                        q_start=q_des
                    )

                    phase = PHASE_BOUNCE

                    phase_start = time.perf_counter()

            # BOUNCE

            elif phase == PHASE_BOUNCE:

                bounce.step(dt)

                if elapsed > MAX_BOUNCE_TIME:

                    print(
                        "[Bounce] Timeout — "
                        "returning to balance"
                    )

                    balance = BallBalanceController(
                        odrvs,
                        motors,
                        phi0,
                        t0,
                        tracker,
                        frame_queue,
                        balance=False
                    )

                    phase = PHASE_BALANCE

                    phase_start = time.perf_counter()


            t_log.append(time.perf_counter() - t0)

            pos_log.append(tracker.pos.copy())
            pos_raw_log.append(tracker.pos_raw.copy())
            vel_log.append(tracker.vel.copy())
            vel_raw_log.append(tracker.vel_raw.copy())

            q_log.append(q_des.copy())


            if tracker.p_hit is not None:
                p_hit_log.append(tracker.p_hit.copy())
            else:
                p_hit_log.append([np.nan, np.nan])

            time.sleep(0)

    # CLEAN EXIT

    except KeyboardInterrupt:

        print("Stopped")

    except Exception as e:

        print(type(e).__name__, e)

    finally:

        motor.hard_stop(motors)

        cv2.destroyAllWindows()

        np.savez(
            "logs/ball_manipulation/run_0.npz",

            t=np.array(t_log),

            ball_pos=np.array(pos_log),
            pos_raw=np.array(pos_raw_log),
            ball_vel=np.array(vel_log),
            vel_raw=np.array(vel_raw_log),

            p_hit=np.array(p_hit_log),

            q=np.array(q_log),
        )



def ping_pong_bot(odrvs, motors, phi0):

    (
        t0,
        tracker,
        frame_queue,
        stop_event,
        cam_thread
    ) = initialize_tracking(motors, phi0)

    # MAIN CONTROL LOOP

    try:

        run_master_loop(
            odrvs,
            motors,
            phi0,
            t0,
            tracker,
            frame_queue
        )

    finally:

        stop_event.set()

        cam_thread.join()






def balancing_bot(odrvs, motors, phi0):

    (
        t0,
        tracker,
        frame_queue,
        stop_event,
        cam_thread
    ) = initialize_tracking(motors, phi0)

    # CONTROLLER

    balance = BallBalanceController(
        odrvs,
        motors,
        phi0,
        t0,
        tracker,
        frame_queue=frame_queue,
        balance=True      # <-- follows geom.get_q_des(t)
    )

    # LOGGING

    t_log = []

    ball_pos_log = []
    ball_pos_raw_log = []

    ball_vel_log = []
    ball_vel_raw_log = []

    ref_log = []
    vel_ref_log = []

    q_log = []

    # MAIN LOOP

    prev_start = time.perf_counter()

    try:

        while True:

            loop_start = time.perf_counter()

            dt = loop_start - prev_start

            prev_start = loop_start

            # MOTOR SAFETY

            check_motor_states(odrvs, motors)

            # DISPLAY CAMERA FRAME

            if show_latest_frame(frame_queue):
                break

            # CONTROLLER STEP

            q_des, ball_ref, ball_vel_ref = balance.step(dt)

            # LOGGING

            t_log.append(loop_start - t0)

            ball_pos_log.append(
                tracker.pos.copy()
            )

            ball_pos_raw_log.append(
                tracker.pos_raw.copy()
            )

            ball_vel_log.append(
                tracker.vel.copy()
            )

            ball_vel_raw_log.append(
                tracker.vel_raw.copy()
            )

            ref_log.append(
                ball_ref.copy()
            )

            vel_ref_log.append(
                ball_vel_ref.copy()
            )

            q_log.append(
                q_des.copy()
            )

            time.sleep(0)

    # CLEAN EXIT

    except KeyboardInterrupt:

        print("Stopped")

    except Exception as e:

        print(type(e).__name__, e)

    finally:

        motor.hard_stop(motors)

        cv2.destroyAllWindows()

        stop_event.set()

        cam_thread.join()

        # SAVE LOG

        np.savez(

            "logs/ball_manipulation/balance_run.npz",

            t=np.array(t_log),

            ball_pos=np.array(ball_pos_log),
            pos_raw=np.array(ball_pos_raw_log),

            ball_vel=np.array(ball_vel_log),
            vel_raw=np.array(ball_vel_raw_log),

            ref_pos=np.array(ref_log),
            ref_vel=np.array(vel_ref_log),

            q=np.array(q_log),
        )
