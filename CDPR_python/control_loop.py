import motor_actions as motor
import geometry as geom
import parameters as p
import utils as utils
import trajectory_planner as traj
import numpy as np
import matplotlib.pyplot as plt
import time
import threading
import mouse
from odrive.utils import dump_errors
from odrive.enums import *
from collections import deque


def run_hybrid_control_loop(odrvs, motors, phi0, Kt):
    print("Starting hybrid control loop")
    print("Press Ctrl+C to stop")

    d_0 = geom.inverse_kinematics([0, 0, 0], p.a, p.b)
    d_des = d_0.copy()

    target_dt = 0.02

    torque_ramp_start_time = None
    torque_ramp_duration = 0.05  # seconds
    torque_ramp_start_factor = 1.5

    q_des_log = []
    d_des_log = []
    d_log = []
    vel_log = []
    torque_log = []
    voltage_log = []
    force_log = []

    t_log = []
    dt_log = []

    d_sample = np.zeros(4)
    vel_sample = np.zeros(4)
    torque_sample = np.zeros(4)

    try:
        motor.position_control(motors)
        time.sleep(0.1)

        print("Moving platform to initial desired pose...")

        q0, q_dot0 = geom.get_q_des(0.0)
        smooth_move_to_pose(motors, phi0, q0, T_MOVE=3.0, dt=0.01)

        print("Initial pose reached. Starting tracking.")

        j_force_current = 0
        sigma = 0

        t0 = time.perf_counter()
        next_t = time.perf_counter()

        while True:
            loop_start = time.perf_counter()
            t = loop_start - t0

            # ---- Safety check ----
            for i, axis in motors.items():
                if axis.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                    dump_errors(odrvs[i])
                    raise Exception(f"Motor {i} not in CLOSED LOOP CONTROL")

            # ---- Trajectory ----
            q_des, q_dot_des = geom.get_q_des(t)
            d_abs = geom.inverse_kinematics(q_des, p.a, p.b)

            for i in range(4):
                d_des[i] = d_0[i] - d_abs[i]
                d_des[i] = traj.voltage_regulator(odrvs[i], d_des[i])

            phi_des = np.array(geom.phi_from_d(d_des, phi0))

            # ---- Hybrid cable selection ----
            j_force, sigma = utils.select_force_controlled_cables(
                p.a, p.b, q_des, j_force_current, p.gamma
            )

            if j_force != j_force_current:
                # set new force motor
                axis_force = motors[j_force]
                axis_force.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
                axis_force.controller.config.input_mode = INPUT_MODE_PASSTHROUGH

                # set previous force motor back to position
                prev_axis = motors[j_force_current]
                prev_axis.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
                prev_axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH

                j_force_current = j_force

                torque_ramp_start_time = time.perf_counter()

            # ---- Always define roles ----
            position_motors = [i for i in range(4) if i != j_force_current]
            axis_force = motors[j_force_current]

            # ---- position control ----
            J = geom.cable_jacobian(q_des, p.a, p.b)
            cable_vel_des = - J @ q_dot_des

            for i in position_motors:
                axis = motors[i]
                motor_vel_des = cable_vel_des[i]*p.motor_signs[i]/(2.0*p.r_d*np.pi)
                axis.controller.input_vel = motor_vel_des
                axis.controller.input_pos = phi_des[i]

            # ---- Torque control ----
            if torque_ramp_start_time is None:
                alpha = 1.0
            else:
                t_ramp = time.perf_counter() - torque_ramp_start_time

                if t_ramp >= torque_ramp_duration:
                    alpha = 1.0
                    torque_ramp_start_time = None  # ramp finished
                else:
                    # linear ramp from 0.2 → 1.0
                    alpha = torque_ramp_start_factor + \
                            (1.0 - torque_ramp_start_factor) * (t_ramp / torque_ramp_duration)
                    
            motor.set_motor_torque(j_force_current, axis_force, alpha*p.f_c[j_force_current])

            # ---- Logging ----
            for i in range(4):
                axis = motors[i]
                d_sample[i] = d_0[i] + utils.cable_length_from_encoder(
                    axis.pos_estimate, phi0, i
                )
                vel_sample[i] = (
                    axis.vel_estimate * p.motor_signs[i] * 2 * np.pi * p.r_d
                )
                torque_sample[i] = axis.motor.foc.Iq_measured * Kt

            t_log.append(t)
            d_log.append(d_sample.copy())
            #d_des_log.append(d_des.copy())
            q_des_log.append(q_des)
            vel_log.append(vel_sample.copy())
            torque_log.append(torque_sample.copy())
            voltage_log.append([odrv.vbus_voltage for odrv in odrvs.values()])
            force_log.append(j_force_current)

            # ---- Hybrid timing ----
            """while True:
                now = time.perf_counter()
                if now >= next_t:
                    break
                if next_t - now > 0.001:
                    time.sleep(0)"""

            next_t += target_dt
            dt_log.append(time.perf_counter() - loop_start)

    except KeyboardInterrupt:
        print("\nHybrid control loop stopped by user")

    except Exception as e:
        print("\nEXCEPTION IN HYBRID CONTROL LOOP:")
        print(type(e).__name__, e)

    finally:
        motor.hard_stop(motors)

        q_est_log = []
        q_prev = q_des_log[0]

        for k in range(len(t_log)):
            q_est, qdot_est = geom.direct_kinematics(
                d_log[k],
                vel_log[k],
                p.a,
                p.b,
                q_prev
            )

            d_des_log.append(geom.inverse_kinematics(q_des_log[k], p.a, p.b))
            q_est_log.append(q_est)
            q_prev = q_est

        q_est_log = np.array(q_est_log)

        utils.save_experiment_data(
            "triangle_hybrid_03",
            t_log,
            d_des_log,
            d_log,
            q_des_log,
            q_est_log,
            torque_log,
            voltage_log,
            force_log=force_log,
            dt_log=dt_log
        )
        utils.plot_trajectory(
            t_log,
            d_des_log,
            d_log,
            q_des_log,
            q_est_log,
            torque_log,
            voltage_log,
            force_log,
            dt_log=dt_log
        )



def run_position_control_loop(odrvs, motors, phi0, Kt):
    print("Starting control loop")
    print("Press Ctrl+C to stop")


    d_0 = geom.inverse_kinematics([0,0,0],p.a,p.b)
    d_des = d_0.copy()

    q_des_log = []
    d_des_log = []
    d_log = []
    vel_log = []
    torque_log = []
    voltage_log = []

    t_log = []
    dt_log = []

    d_sample = np.zeros(4)
    vel_sample = np.zeros(4)
    torque_sample = np.zeros(4)

    compute_time = 0
    read_tot = 0
    send_tot = 0
    append_tot = 0

    read_log = []
    send_log = []
    append_time_log = []
    compute_log = []


    try:
        motor.position_control(motors)
        time.sleep(0.1)

        print("Moving platform to initial desired pose...")

        q0, q_dot0 = geom.get_q_des(0.0)

        smooth_move_to_pose(motors, phi0, q0, T_MOVE=3.0, dt=0.01)

        print("Initial pose reached. Starting tracking.")

        loop_count = 0
        t0 = time.perf_counter()
        loop_start = time.perf_counter()
        
        while True:
            loop_count+=1
            t = time.perf_counter()-t0
            loop_start = time.perf_counter()

            if loop_count % 50 == 0:
                ti = time.perf_counter()
                for i, axis in motors.items():
                    if axis.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                        dump_errors(odrvs[i])
                        raise Exception(f"Motor {i} not in CLOSED LOOP CONTROL")
                read = time.perf_counter()-ti
            else:
                read = 0


            q_des, q_dot_des = geom.get_q_des(t)
            d_abs = geom.inverse_kinematics(q_des,p.a,p.b)

            ti = time.perf_counter()
            for i in range(4):
                d_des[i] = d_0[i] - d_abs[i]
                d_des[i] = traj.voltage_regulator(odrvs[i], d_des[i])

                axis = motors[i]
                d_sample[i] = d_0[i]+utils.cable_length_from_encoder(
                    axis.pos_estimate, phi0, i
                )

                vel_sample[i] = (
                    axis.vel_estimate * p.motor_signs[i] * 2*np.pi*p.r_d
                )

                torque_sample[i] = axis.motor.foc.Iq_measured * Kt

            voltage_log.append([odrv.vbus_voltage for odrv in odrvs.values()])
            motor_read = time.perf_counter()-ti


            motor_vel_des = motor.velocity_feedforward(motors, q_des, q_dot_des)
            
            phi_des = np.array(geom.phi_from_d(d_des, phi0))

            ti = time.perf_counter()
            for i, axis in motors.items():
                axis.controller.input_vel = motor_vel_des[i]
                axis.controller.input_pos = phi_des[i]
            motor_send = time.perf_counter()-ti
            
            read_tot = motor_read+read
            # ---- ATOMIC APPEND ----
            t_log.append(t)
            d_log.append(d_sample.copy())
            #d_des_log.append(d_des.copy())
            q_des_log.append(q_des)
            vel_log.append(vel_sample.copy())
            torque_log.append(torque_sample.copy())
            read_log.append(read_tot)
            send_log.append(motor_send)


            # ---- timing ----
            """while True:
                now = time.perf_counter()
                if now >= next_t:
                    break
                if next_t - now > 0.001:
                    time.sleep(0)

            next_t += target_dt"""
            

            dt = time.perf_counter() - loop_start
            dt_log.append(dt)
            compute_time = dt-read_tot-motor_send
            compute_log.append(compute_time)




    except KeyboardInterrupt:
        print("\nControl loop stopped by user")

    except Exception as e:
        print("\nEXCEPTION IN CONTROL LOOP:")
        print(type(e).__name__, e)

    
    finally:
        motor.hard_stop(motors)
        q_est_log = []
        d_des_log = []

        q_prev = q_des_log[0]   # good initial guess

        for k in range(len(t_log)):
            d_log_k = d_log[k]
            q_est, qdot_est = geom.direct_kinematics(
                d_log_k,
                vel_log[k],
                p.a,
                p.b,
                q_prev
            )

            d_des = geom.inverse_kinematics(q_des_log[k], p.a, p.b)

            q_est_log.append(q_est)
            d_des_log.append(d_des)
            q_prev = q_est

        q_est_log = np.array(q_est_log)

        utils.save_experiment_data(
            "dt_test_position_00", 
            t_log, 
            d_des_log, 
            d_log, 
            q_des_log, 
            q_est_log, 
            torque_log, 
            voltage_log,
            dt_log=dt_log,
            read_log=read_log,
            send_log=send_log,
            compute_log=compute_log
        )
        utils.plot_trajectory(
            t_log, 
            d_des_log, 
            d_log, 
            q_des_log, 
            q_est_log, 
            torque_log, 
            voltage_log, 
            None, 
            dt_log=dt_log
        )



def run_velocity_control_loop(odrvs, motors, phi0, Kt):
    print("Starting control loop")
    print("Press Ctrl+C to stop")

    Kp = 5
    Kd = 0.1

    d_0 = geom.inverse_kinematics([0,0,0],p.a,p.b)
    d_des = d_0.copy()

    dt = 0.02
    target_dt = 0.02
    next_t = time.perf_counter()

    q_des_log = []
    d_des_log = []
    d_log = []
    vel_log = []
    torque_log = []
    voltage_log = []

    t_log = []
    dt_log = []

    d_des = np.zeros(4)
    d_act = np.zeros(4)

    d_sample = np.zeros(4)
    vel_sample = np.zeros(4)
    torque_sample = np.zeros(4)


    try:
        time.sleep(0.1)

        print("Moving platform to initial desired pose...")

        q0, q_dot0 = geom.get_q_des(0.0)

        smooth_move_to_pose(motors, phi0, q0, T_MOVE=3.0, dt=0.01)

        print("Initial pose reached. Starting tracking.")

        q_prev = q0.copy()

        motor.velocity_control(motors)

        t0 = time.time()
        loop_start = time.perf_counter()
        
        while True:
            t = time.time()-t0
            loop_start = time.perf_counter()

            for i, axis in motors.items():
                if axis.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                    dump_errors(odrvs[i])
                    raise Exception(f"Motor {i} not in CLOSED LOOP CONTROL")


            q_des, q_dot_des = geom.get_q_des(t)
            d_abs = geom.inverse_kinematics(-q_des, p.a, p.b)
            
            for i in range(4):
                d_des[i] = d_0[i] - d_abs[i]
            if not (p.WORKSPACE_X_MIN < q_des[0] < p.WORKSPACE_X_MAX and
                    p.WORKSPACE_Y_MIN < q_des[1] < p.WORKSPACE_Y_MAX):
                raise Exception("Platform out of workspace bounds")

            J = geom.cable_jacobian(q_des, p.a, p.b)
            d_dot_des = -J @ q_dot_des
            
            cable_vels = []
            for i in range(4):
                cable_vels.append(
                    p.motor_signs[i] *
                    motors[i].vel_estimate * 2*np.pi*p.r_d
                )
                d_act[i] = utils.cable_length_from_encoder(motors[i].pos_estimate, phi0, i)
            
            print(d_des, d_act)
            e_d = d_des - d_act
            e_v = d_dot_des - cable_vels

            d_dot_cmd = d_dot_des - Kp * e_d + Kd * e_v 
            d_dot_cmd = np.clip(d_dot_cmd, -1, 1)

            """# --- Direct kinematics ---
            q_est, q_dot_est = geom.direct_kinematics(
                d_act,
                cable_vels,
                p.a,
                p.b,
                q_prev
            )
            q_prev = q_est

            e_q = q_des - q_est
            e_v = q_dot_des - q_dot_est

            q_dot_cmd = q_dot_des + Kp_pos @ e_q + Kd_pos @ e_v

            q_dot_cmd[:2] = np.clip(q_dot_cmd[:2], -1, 1)   # m/s
            q_dot_cmd[2]  = np.clip(q_dot_cmd[2],  -1, 1)   # rad/s

            J = geom.cable_jacobian(q_est, p.a, p.b)
            d_dot_cmd = -J @ q_dot_cmd
            d_dot_cmd = np.clip(d_dot_cmd, -1, 1)   # m/s"""


            for i, odrv in odrvs.items():
                vbus = odrv.vbus_voltage
                if vbus > 25:
                    d_dot_cmd[i] += (vbus - 25) * 0.2 * np.sign(d_dot_cmd[i])


            for i, axis in motors.items():
                motor_vel = (
                    d_dot_cmd[i]
                    * p.motor_signs[i]
                    / (2 * np.pi * p.r_d)
                )
                axis.controller.input_vel = motor_vel


            for i in range(4):
                axis = motors[i]
                d_sample[i] = -utils.cable_length_from_encoder(
                    axis.pos_estimate, phi0, i
                )
                vel_sample[i] = (
                    axis.vel_estimate * p.motor_signs[i] * 2*np.pi*p.r_d
                )
                torque_sample[i] = axis.motor.foc.Iq_measured * Kt

            # ---- ATOMIC APPEND ----
            t_log.append(t)
            d_log.append(d_sample.copy())
            d_des_log.append(d_des.copy())
            q_des_log.append(q_des)
            vel_log.append(vel_sample.copy())
            torque_log.append(torque_sample.copy())
            voltage_log.append([odrv.vbus_voltage for odrv in odrvs.values()])
            dt = time.perf_counter() - loop_start
            dt_log.append(dt)

            # ---- timing ----
            next_t += target_dt
            sleep_time = next_t - time.perf_counter()

            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_t = time.perf_counter()




    except KeyboardInterrupt:
        print("\nControl loop stopped by user")

    except Exception as e:
        print("\nEXCEPTION IN CONTROL LOOP:")
        print(type(e).__name__, e)

    
    finally:
        motor.hard_stop(motors)
        q_est_log = []

        q_prev = q_des_log[0]   # good initial guess

        for k in range(len(t_log)):
            d_log_k = d_0 - d_log[k]
            q_est, qdot_est = geom.direct_kinematics(
                d_log_k,
                vel_log[k],
                p.a,
                p.b,
                q_prev
            )

            q_est_log.append(q_est)
            q_prev = q_est

        q_est_log = np.array(q_est_log)
        utils.plot_trajectory(t_log, d_des_log, d_log, q_des_log, q_est_log, torque_log, voltage_log, None, dt_log=dt_log)


def run_keyboard_control_loop(odrvs, motors, phi0, Kt):
    print("Starting control loop")
    print("Press Ctrl+C to stop")

    input_mode = int(input("Press 0 for keyboard control, 1 for mouse control: "))

    d_0 = geom.inverse_kinematics([0,0,0], p.a, p.b)

    d = []
    for i, axis in motors.items():
        d.append(
            d_0[i] + utils.cable_length_from_encoder(
                axis.pos_estimate, phi0, i
            )        
        )

    d_des = d.copy()

    target_dt = 0.02

    q_des_log = []
    d_des_log = []
    d_log = []
    vel_log = []
    torque_log = []
    voltage_log = []

    t_log = []
    dt_log = []

    d_sample = np.zeros(4)
    vel_sample = np.zeros(4)
    torque_sample = np.zeros(4)

    screen_center = mouse.get_position()


    try:
        motor.position_control(motors)
        time.sleep(0.1)

        q_des, q_dot_est = geom.direct_kinematics(d, [0,0,0,0], p.a, p.b, [0,0,0])

        t0 = time.perf_counter()
        next_t = time.perf_counter()
        loop_start = time.perf_counter()
        
        while True:
            t = time.perf_counter()-t0
            loop_start = time.perf_counter()

            for i, axis in motors.items():
                if axis.current_state != AXIS_STATE_CLOSED_LOOP_CONTROL:
                    dump_errors(odrvs[i])
                    raise Exception(f"Motor {i} not in CLOSED LOOP CONTROL")

            q_des = geom.input_q_des(q_des, input_mode, screen_center)
            
            d_abs = geom.inverse_kinematics(q_des,p.a,p.b)

            for i in range(4):
                d_des[i] = d_0[i] - d_abs[i]
                d_des[i] = traj.voltage_regulator(odrvs[i], d_des[i])

                axis = motors[i]
                d_sample[i] = d_0[i]+utils.cable_length_from_encoder(
                    axis.pos_estimate, phi0, i
                )
                vel_sample[i] = (
                    axis.vel_estimate * p.motor_signs[i] * 2*np.pi*p.r_d
                )
                torque_sample[i] = axis.motor.foc.Iq_measured * Kt

            #motor.velocity_feedforward(motors, q_des, q_dot_des)
            
            phi_des = np.array(geom.phi_from_d(d_des, phi0))

            for i, axis in motors.items():
                axis.controller.input_pos = phi_des[i]
            

            # ---- ATOMIC APPEND ----
            t_log.append(t)
            d_log.append(d_sample.copy())
            #d_des_log.append(d_des.copy())
            q_des_log.append(q_des.copy())
            vel_log.append(vel_sample.copy())
            torque_log.append(torque_sample.copy())
            voltage_log.append([odrv.vbus_voltage for odrv in odrvs.values()])

            # ---- timing ----
            while True:
                now = time.perf_counter()
                if now >= next_t:
                    break
                if next_t - now > 0.001:
                    time.sleep(0)

            next_t += target_dt

            dt = time.perf_counter() - loop_start
            dt_log.append(dt)




    except KeyboardInterrupt:
        print("\nControl loop stopped by user")

    except Exception as e:
        print("\nEXCEPTION IN CONTROL LOOP:")
        print(type(e).__name__, e)

    
    finally:
        motor.hard_stop(motors)
        q_est_log = []
        d_des_log = []

        q_prev = q_des_log[0]   # good initial guess

        for k in range(len(t_log)):
            d_log_k = d_log[k]
            q_est, qdot_est = geom.direct_kinematics(
                d_log_k,
                vel_log[k],
                p.a,
                p.b,
                q_prev
            )

            d_des = geom.inverse_kinematics(q_des_log[k], p.a, p.b)

            q_est_log.append(q_est)
            d_des_log.append(d_des)
            q_prev = q_est

        q_est_log = np.array(q_est_log)
        utils.plot_trajectory(t_log, d_des_log, d_log, q_des_log, q_est_log, torque_log, voltage_log, None, dt_log=dt_log)



def user_move_to_pose(motors, phi0, T_MOVE=3.0):
    """
    Ask the user for a desired pose and move the platform there smoothly.
    """

    try:
        print("Enter desired platform pose:")
        x = float(input("  x [m]: "))
        y = float(input("  y [m]: "))
        theta = float(input("  theta [rad]: "))
    except ValueError:
        print("Invalid input. Aborting move.")
        return

    q_target = [x, y, theta]

    print(f"Moving platform to pose: {q_target}")
    smooth_move_to_pose(
        motors,
        phi0,
        q_target,
        T_MOVE=T_MOVE
    )

    print("Move complete.")


def ramp_alpha(t, T):
    s = np.clip(t / T, 0.0, 1.0)
    return 3*s**2 - 2*s**3

def smooth_move_to_pose(motors, phi0, q_target, T_MOVE=3.0, dt=0.01):
    """
    Smoothly move the platform from current pose to q_target.

    Parameters
    ----------
    motors : dict {int: axis}
        Motor axes
    d_init : list (4,)
        Motor zero offsets (turns)
    q_target : array-like (3,)
        Desired pose [x, y, theta]
    T_MOVE : float
        Move duration [s]
    dt : float
        Control timestep [s]
    """

    q_target = np.asarray(q_target, dtype=float)

    # --- ensure position control ---
    motor.position_control(motors)
    time.sleep(0.1)

    # --- reference cable lengths at origin ---
    d_0 = geom.inverse_kinematics([0, 0, 0], p.a, p.b)

    # --- estimate current pose from cables ---
    d_start = []
    for i, axis in motors.items():
        d_start.append(
            d_0[i] + utils.cable_length_from_encoder(
                axis.pos_estimate, phi0, i
            )
        )

    q_start, _ = geom.direct_kinematics(
        d_start,
        [0, 0, 0, 0],
        p.a,
        p.b,
        q_prev=[0, 0, 0]
    )
    print("q_start:", q_start)
    print("q_target:", q_target)

    # --- smooth interpolation loop ---
    t0 = time.time()
    while True:
        t = time.time() - t0
        if t >= T_MOVE:
            break

        alpha = ramp_alpha(t, T_MOVE)
        q_ref = q_start + alpha * (q_target - q_start)

        d_abs = geom.inverse_kinematics(q_ref, p.a, p.b)
        d_ref = d_0 - d_abs
        phi_ref = geom.phi_from_d(d_ref, phi0)

        for i, axis in motors.items():
            axis.controller.input_pos = phi_ref[i]

        time.sleep(dt)

    # --- final hold (exact target) ---
    d_abs = geom.inverse_kinematics(q_target, p.a, p.b)
    d_ref = d_0 - d_abs
    phi_ref = geom.phi_from_d(d_ref, phi0)

    for i, axis in motors.items():
        axis.controller.input_pos = phi_ref[i]




def taut_observer_mode(motors, phi0, tension = 0.2):
    print("Starting taut-cable pose observer mode")
    print("Press Ctrl+C to stop")

    d_0 = geom.inverse_kinematics([0, 0, 0], p.a, p.b)

    q_prev = [0, 0, 0]

    try:

        utils.init_live_cdpr_plot()

        for i, axis in motors.items():
            axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
            axis.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
            axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
        
        
        while True:
            

            d = []
            for i, axis in motors.items():
                motor.set_motor_torque(i, axis, tension)
                d.append(
                    d_0[i] + utils.cable_length_from_encoder(
                        axis.pos_estimate, phi0, i
                    )        
                )

            v = []
            for i, axis in motors.items():
                v.append(
                    p.motor_signs[i] *
                    axis.vel_estimate * 2*np.pi*p.r_d
                )

            q, vel = geom.direct_kinematics(
                d,
                v,
                p.a,
                p.b,
                q_prev
            )

            utils.update_live_cdpr_plot(q, d)

            q_prev = q
    
    except KeyboardInterrupt:
        print("\nObserver mode stopped by user")
    
    finally:
        motor.hard_stop(motors)
        plt.close("all")



traj_buffer = deque(maxlen=300)
traj_lock = threading.Lock()

log_buffer = []
log_lock = threading.Lock()

executor_dt_buf = np.zeros(500)
executor_dt_idx = 0
executor_dt_lock = threading.Lock()


def run_multithred_control(odrvs, motors, phi0, Kt):

    stop_event = threading.Event()

    d_0 = geom.inverse_kinematics(np.array([0, 0, 0]), p.a, p.b)

    q0 = geom.get_q_des(0)
    print(q0)
    smooth_move_to_pose(motors, phi0, q0)

    planner = threading.Thread(
        target=planner_thread,
        args=(odrvs, motors, phi0, d_0, stop_event),
        daemon=True
    )

    executor = threading.Thread(
        target=executor_loop,
        args=(motors, stop_event),
        daemon=True
    )

    planner.start()
    executor.start()

    try:
        while True:
            time.sleep(0.1)   # main thread idle
    except KeyboardInterrupt:
        stop_event.set()
        planner.join()
        executor.join()
        motor.hard_stop(motors)

    with executor_dt_lock:
        count = executor_dt_idx
        buf = executor_dt_buf.copy()

    with log_lock:
        logs = list(log_buffer)

    t_log = np.array([l["t"] for l in logs])
    q_des_log = np.array([l["q_des"] for l in logs])
    d_des_log = np.array([l["d_des"] for l in logs])
    phi_act_log = np.array([l["phi_act"] for l in logs])
    vbus_log = np.array([l["vbus"] for l in logs])
    phi_dot_log = np.array([l["phi_dot"] for l in logs])
    torque_log = np.array([l["torque"] for l in logs])
    dt_log = np.array([l["dt"] for l in logs])
    exec_dt_log = np.array(["dt_exec" for l in logs])
    
    d_act_log = []
    vel_log = []
    q_est_log = []
    q_prev = q_des_log[0]

    d_act_sample = np.zeros(4)
    vel_sample = np.zeros(4)
    for k in range(len(t_log)):
        for i in range(4):
            d_act_sample[i] = d_0[i]+utils.cable_length_from_encoder(phi_act_log[k,i], phi0, i)
            vel_sample[i] = phi_dot_log[k,i]*2*np.pi*p.r_d*p.motor_signs[i]
        d_act_log.append(d_act_sample.copy())
        vel_log.append(vel_sample.copy())

        q_est, q_dot_est = geom.direct_kinematics(d_act_log[k], vel_log[k], p.a, p.b, q_prev)
        q_est_log.append(q_est)
        q_prev = q_est

    N = len(buf)

    if count < N:
        exec_dt_log = buf[:count]
    else:
        start = count%N
        exec_dt_log = np.concatenate((buf[start:], buf[:start]))


    utils.plot_trajectory(t_log, d_des_log, d_act_log, q_des_log, q_est_log, torque_log, vbus_log, None, dt_log, exec_dt_log)



def planner_thread(odrvs, motors, phi0, d_0, stop_event):
    t_traj = 0.0
    t_start = time.perf_counter()
    last_time = time.perf_counter()
    dt_plan = 0.02
    horizon = 0.3   # seconds into the future

    phi_sample = np.zeros(4)
    vel_sample = np.zeros(4)
    torque_sample = np.zeros(4)
    vbus = np.zeros(4)
    phi_dot_des = np.zeros(4)


    while not stop_event.is_set():
        now = time.perf_counter()
        dt = now - last_time
        last_time = now

        t_traj = now - t_start

        q_des_now, q_dot_des_now = geom.get_q_des(t_traj)
        d_des_now = geom.inverse_kinematics(q_des_now, p.a, p.b)

        new_points = []

        with traj_lock:
            if len(traj_buffer) > 0:
                t_plan_start = max(traj_buffer[-1][0], now)
                last_t = traj_buffer[-1][0]
            else:
                t_plan_start = now
                last_t = now-dt_plan

        # compute future trajectory
        t_local = max(t_plan_start, last_t + dt_plan)
        while t_local < now + horizon:
            q_des, q_dot_des = geom.get_q_des(t_local - t_start)
            d_abs = geom.inverse_kinematics(q_des, p.a, p.b)

            d_des = d_0 - d_abs
            phi_des = np.array(geom.phi_from_d(d_des, phi0))

            J = geom.cable_jacobian(q_des, p.a, p.b)
            d_dot_des = -J @ q_dot_des

            for i in range(4):
                phi_dot_des[i] = (
                    d_dot_des[i]
                    * p.motor_signs[i]
                    / (2*np.pi*p.r_d)
                    )

            new_points.append((t_local, phi_des, phi_dot_des))
            t_local += dt_plan

        # replace buffer
        with traj_lock:
            for pt in new_points:
                traj_buffer.append(pt)

        for i, axis in motors.items():
            phi_sample[i] = axis.pos_estimate
            vel_sample[i] = axis.vel_estimate
            torque_sample[i] = axis.motor.foc.Iq_measured
            vbus[i] = odrvs[i].vbus_voltage
        
        with executor_dt_lock:
            dt_snapshot = executor_dt_buf.copy()

        with log_lock:
            log_buffer.append({
                "t": t_traj,               
                "q_des": q_des_now.copy(),
                "d_des": d_des_now.copy(),
                "phi_act": phi_sample.copy(),
                "vbus": vbus.copy(),
                "phi_dot": vel_sample.copy(),
                "torque": torque_sample.copy(),
                "dt": dt,
                "dt_exec": dt_snapshot
            })

        time.sleep(0.01)


def executor_loop(motors, stop_event):
    last_time = time.perf_counter()
    global executor_dt_idx
    print_time = time.perf_counter()

    motor.position_control(motors)

    while not stop_event.is_set():
        now = time.perf_counter()
        
        dt = now - last_time
        last_time = now

        if now - print_time > 1:
            print(
                "now:", now,
                "buf_start:", traj_buffer[0][0],
                "buf_end:", traj_buffer[-1][0],
                "span:", traj_buffer[-1][0] - traj_buffer[0][0]
            )
            print_time = now



        with executor_dt_lock:
            executor_dt_buf[executor_dt_idx % len(executor_dt_buf)] = dt
            executor_dt_idx += 1

        with traj_lock:
            if len(traj_buffer) < 2:
                # nothing planned yet → hold last command
                time.sleep(0)
                continue


            while traj_buffer[1][0] < now:
                traj_buffer.popleft()

            # find two surrounding points
            t0, phi0, phi_dot0 = traj_buffer[0]
            t1, phi1, phi_dot1 = traj_buffer[1]
            

        # simple linear interpolation
        if t1 > t0:
            alpha = np.clip((now - t0) / (t1 - t0), 0.0, 1.0)
        else:
            alpha = 0.0

        print(alpha)
        phi_cmd = (1 - alpha) * phi0 + alpha * phi1
        phi_dot_cmd = (1 - alpha) * phi_dot0 + alpha * phi_dot1

        print(f"phi_des: ", phi0, "phi_cmd: ", phi_cmd)

        # send to motors
        for i, axis in motors.items():
            axis.controller.input_vel = phi_dot_cmd[i]
            axis.controller.input_pos = phi_cmd[i]

        time.sleep(0)
