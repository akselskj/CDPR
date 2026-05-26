import gc
import os
import time

import numpy as np
import psutil

import control_loop as ctrl
import geometry as geom
import motor_actions as motor
import parameters as p
import ping_pong_bot as ping


"""
Connect to the motor drivers and provide a menu for running robot modes.
"""


def print_menu():
    print("\n=== CDPR Main Menu ===")
    print("1 : Initialize / tighten cables")
    print("2 : Run position control loop")
    print("3 : Run hybrid control loop")
    print("4 : Run Ping Pong demo")
    print("5 : Run velocity control loop")
    print("6 : HARD STOP (IDLE motors)")
    print("7 : move to home")
    print("8 : Move to user-defined pose")
    print("9 : Taut-cable pose observer (torque + live DK)")
    print("10 : Clear ODrive errors")
    print("11 : multithread position loop")
    print("12 : Run user control mode")
    print("13 : Run ball balancing")
    print("exit : Exit program")


def configure_runtime():
    gc.disable()

    process = psutil.Process(os.getpid())
    process.nice(0)


def estimate_home_encoder_offsets(motors):
    d_ref = geom.inverse_kinematics([0, 0, 0], p.a, p.b)
    d_home = geom.inverse_kinematics(p.home, p.a, p.b)
    delta_d = d_ref - d_home

    phi0 = []
    for i, axis in motors.items():
        offset = (
            axis.pos_estimate
            -
            delta_d[i] * p.motor_signs[i] / (2 * np.pi * p.r_d)
        )
        phi0.append(offset)

    return phi0


def main():
    configure_runtime()

    motors, odrvs = motor.discover_motors()
    print("CDPR control started")

    phi0 = estimate_home_encoder_offsets(motors)
    Kt = motor.torque_constant(motors)

    while True:

        print_menu()
        user_input = input("Select option: ").strip()

        if user_input == "exit":
            print("Exiting program...")
            motor.hard_stop(motors)
            break

        elif user_input.isdigit():
            cmd = int(user_input)

            if cmd == 1:
                phi0 = motor.init_tension(motors, 0.2)
                motor.print_motor_positions(motors)
                print("phi0 = ", phi0)

            elif cmd == 2:
                ctrl.run_position_control_loop(odrvs, motors, phi0, Kt)

            elif cmd == 3:
                ctrl.run_hybrid_control_loop(odrvs, motors, phi0, Kt)

            elif cmd == 4:
                ping.ping_pong_bot(odrvs, motors, phi0)

            elif cmd == 5:
                # This is not working correctly.
                ctrl.run_velocity_control_loop(odrvs, motors, phi0, Kt)

            elif cmd == 6:
                motor.hard_stop(motors)

            elif cmd == 7:
                ctrl.smooth_move_to_pose(motors, phi0, p.home)
                time.sleep(0.5)
                motor.hard_stop(motors)

            elif cmd == 8:
                ctrl.user_move_to_pose(motors, phi0)
                time.sleep(0.5)
                motor.hard_stop(motors)

            elif cmd == 9:
                # Direct kinematics test: move the platform manually and compare with the plot.
                ctrl.taut_observer_mode(motors, phi0, 0.2)

            elif cmd == 10:
                motor.clear_all_errors(odrvs)

            elif cmd == 11:
                # USB communication is currently the bottleneck, but this may be useful later.
                ctrl.run_multithred_control(odrvs, motors, phi0, Kt)

            elif cmd == 12:
                ctrl.run_keyboard_control_loop(odrvs, motors, phi0, Kt)

            elif cmd == 13:
                ping.balancing_bot(odrvs, motors, phi0)

            else:
                print("Unknown command")

        else:
            print("Invalid input")


if __name__ == "__main__":
    main()
