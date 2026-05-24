import odrive
import time
import parameters as p
import numpy as np
import geometry as geom
from odrive.enums import *
from odrive.utils import dump_errors
import sys
import os

if os.name == "nt":                # Windows
    import msvcrt
    def kb_hit():
        return msvcrt.kbhit()
    def get_key():
        return msvcrt.getch()
else:                              # POSIX (Linux, macOS, …)
    import termios, tty, select

    def kb_hit():
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        return bool(dr)

    def get_key():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)
        return ch.encode()          # keep same API as msvcrt.getch()


def discover_motors():
    SERIAL_TO_MOTOR = {
        "394A353B3231": 0,
        "393E353C3231": 1,
        "393D35443231": 2,
        "394A35513231": 3,
    }

    axes = {}
    odrvs = {}

    print("Connecting to ODrives by serial number...")

    for serial, motor_id in SERIAL_TO_MOTOR.items():
        print(f"Connecting to motor {motor_id} (serial {serial})")
        odrv = odrive.find_any(serial_number=serial, timeout=20)
        axis = odrv.axis0
        odrv.config.dc_bus_overvoltage_trip_level = 34
        odrv.config.brake_resistor0.resistance = 2.0
        odrv.config.brake_resistor0.enable = True
        odrv.config.brake_resistor0.enable_dc_bus_voltage_feedback = True
        odrv.config.brake_resistor0.dc_bus_voltage_feedback_ramp_start = 26
        odrv.config.brake_resistor0.dc_bus_voltage_feedback_ramp_end = 30
        axis.config.anticogging.enabled = True
        axis.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
        axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
        axis.controller.config.vel_limit = 50.0   # turns per second
        axis.controller.config.vel_limit_tolerance = 1.2
        axis.config.motor.current_hard_max = 40.0
        axis.config.motor.current_soft_max = 20.0
        axis.requested_state = AXIS_STATE_IDLE
        axis.pos_estimate = odrv.rs485_encoder_group0.raw
        print(f"Motor {motor_id} set to IDLE")
        axes[motor_id] = axis
        odrvs[motor_id] = odrv

    time.sleep(0.2)
    return axes, odrvs

def hard_stop(motors):
    print("HARD STOP: setting all motors to IDLE")
    for i, axis in motors.items():
        axis.requested_state = AXIS_STATE_IDLE
        print(f"Motor {i}: IDLE")

def torque_constant(motors):
    return motors[0].config.motor.torque_constant
    
def position_control(motors, index = np.array([0, 1, 2, 3])):
    print("Setting motors to CLOSED LOOP CONTROL")
    for i in index:
        axis = motors[i]
        axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        axis.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
        axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
        print(f"Motor {i}: CLOSED LOOP CONTROL")


def velocity_control(motors, index = np.array([0, 1, 2, 3])):
    print("Setting motors to CLOSED LOOP CONTROL")
    for i in index:
        axis = motors[i]
        axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL
        axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
        print(f"Motor {i}: VELOCITY CONTROL")


def init_tension(motors, tension=0.2):
    """
    Apply small holding torque to tighten cables
    Platform must be mechanically fixed!
    """
    print("Initializing cable tension")
    print("ENSURE NO cables are slack")

    confirm = input("Type 'y' to continue: ").strip().lower()
    if confirm != "y":
        print("Aborted")
        return

    print("Type 'y' and press Enter to stop.")

    for i, axis in motors.items():
        axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
        axis.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
        axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH

    while True:
        for i, axis in motors.items():
            set_motor_torque(i,axis, tension) #tension/3 if (i == 1 or i == 2) else tension)

        if kb_hit():
            key = get_key()
            if key == b'y':
                break

        #time.sleep(dt)
    
    d_ref = geom.inverse_kinematics([0,0,0], p.a, p.b)
    d_home = geom.inverse_kinematics(p.home, p.a, p.b)
    delta_d = d_ref - d_home
    phi0 = []
    for i, axis in motors.items():
        phi0.append(axis.pos_estimate - delta_d[i]*p.motor_signs[i]/(2*np.pi*p.r_d))
    print(phi0)
    hard_stop(motors)
    return phi0

def print_motor_positions(motors):
    print("Motor positions:")
    for i, axis in motors.items():
        pos = axis.pos_estimate
        print(f"Motor {i}: {pos:.4f} turns")


def set_motor_torque(motor_idx, axis, torque = 0.1):
    axis.controller.input_torque = torque*p.motor_signs[motor_idx]


def velocity_feedforward(axes, q, q_dot_des):
    J = geom.cable_jacobian(q, p.a, p.b)
    cable_vel_des = - J @ q_dot_des
    motor_vel_des = np.zeros(4)
    for i, axis in axes.items():
        motor_vel_des[i] = cable_vel_des[i]*p.motor_signs[i]/(2.0*p.r_d*np.pi)
    return motor_vel_des


def motor_input(axes, q, q_dot_des, phi0):
    J = geom.cable_jacobian(q, p.a, p.b)
    cable_vel_des = - J @ q_dot_des

    d_abs = geom.inverse_kinematics(q, p.a, p.b)
    phi_des = geom.phi_from_d(d_abs, phi0)

    for i, axis in axes.items():
        motor_vel_des = cable_vel_des[i]*p.motor_signs[i]/(2.0*p.r_d*np.pi)
        axis.controller.input_vel = motor_vel_des
        axis.controller.input_pos = phi_des[i]



def clear_all_errors(odrvs):
    for i, odrv in odrvs.items():
        print(f"Clearing errors on ODRV {i}...")
        dump_errors(odrv)
        odrv.clear_errors()
        print(f"ODRV {i} errors cleared")