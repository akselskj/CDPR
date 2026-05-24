import odrive
from odrive.enums import *
import time

SERIAL_TO_MOTOR = {
    "394A353B3231": 0,
    "393E353C3231": 1,
    "393D35443231": 2,
    "394A35513231": 3,
}

odrives = {}
axes = {}

print("Connecting to ODrives by serial number...")

for serial, motor_id in SERIAL_TO_MOTOR.items():
    print(f"Connecting to motor {motor_id} (serial {serial})")
    odrv = odrive.find_any(serial_number=serial, timeout=20)
    axis = odrv.axis0
    axis.controller.config.control_mode = CONTROL_MODE_POSITION_CONTROL
    axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
    axis.controller.config.vel_limit = 5.0   # turns per second
    axis.controller.config.vel_limit_tolerance = 1.2
    axis.config.motor.current_hard_max = 25.0
    axis.config.motor.current_soft_max = 23.0
    axis.requested_state = AXIS_STATE_IDLE
    print(f"Motor {motor_id} set to IDLE")
    axis.pos_estimate = odrv.rs485_encoder_group0.raw
    axis.controller.config.control_mode = CONTROL_MODE_VELOCITY_CONTROL

    axis.controller.config.vel_gain = 0.5
    axis.config.anticogging.max_torque = 0.15
    axis.config.anticogging.calib_start_vel = 0.5
    axis.config.anticogging.calib_end_vel = 0.15
    axis.config.anticogging.calib_coarse_integrator_gain = 10


    axis.requested_state = AXIS_STATE_ANTICOGGING_CALIBRATION
    odrives[motor_id] = odrv
    axes[motor_id] = axis

time.sleep(0.5)

while axes[3].current_state != AXIS_STATE_IDLE:
    time.sleep(0.1)
print("Anticogging calibration complete.")
for i, axis in axes.items():
    axis.requested_state = AXIS_STATE_IDLE
    axis.config.anticogging.enabled = True
    print(f"Motor {i}: Anticogging enabled.")
    print("Motor set to IDLE.")
    time.sleep(0.2)
    print("Saving configuration to ODrive...")
    odrives[i].save_configuration()
    print("Done.")
    time.sleep(0.2)


