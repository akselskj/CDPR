import odrive
import time
import msvcrt
import matplotlib.pyplot as plt
import numpy as np
from odrive.enums import *

SERIAL_TO_MOTOR = {
    "394A353B3231": 0,
    "393E353C3231": 1,
    "393D35443231": 2,
    "394A35513231": 3,
}

# -------- Select motor --------
motor_id = 2
serial = next(s for s, m in SERIAL_TO_MOTOR.items() if m == motor_id)

print(f"Connecting to motor {motor_id} (serial {serial})")
odrv = odrive.find_any(serial_number=serial, timeout=20)
axis = odrv.axis0

axis.config.motor.current_hard_max = 15.0
axis.config.motor.current_soft_max = 13.0


# -------- Enter closed loop & torque mode --------
axis.requested_state = AXIS_STATE_CLOSED_LOOP_CONTROL
axis.controller.config.control_mode = CONTROL_MODE_TORQUE_CONTROL
axis.controller.config.input_mode = INPUT_MODE_PASSTHROUGH
axis.controller.config.vel_limit = 2   # turns per second
axis.controller.config.vel_limit_tolerance = 1.2



time.sleep(0.2)

print("Motor in torque CONTROL")
print(f"Current position: {axis.pos_estimate:.4f} turns")

q0 = axis.pos_estimate

# --------- For plotting ---------
t0 = time.time()
t_log = []
pos_log = []
vel_log = []


t = time.time()-t0

# -------- Interactive loop --------
try:
    while t<=30:
        t = time.time()-t0

        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'q':
                print("\nEmergency stop (q pressed)")
                break
        
        axis.controller.input_torque = 1

        t_log.append(t)
        pos_log.append(axis.pos_estimate-q0)
        vel_log.append(axis.vel_estimate)


except KeyboardInterrupt:
    print("\nInterrupted by user.")

finally:
    axis.requested_state = AXIS_STATE_IDLE
    print("Motor set to IDLE.")



"""
fig, ax = plt.subplots()

ax.plot(t_log, pos_log, label='Measured Position')
ax.xaxis.set_label_text('Time [s]')
ax.yaxis.set_label_text('Position [turns]')
ax.grid()
plt.legend()
plt.show()

fig, ax = plt.subplots()

ax.plot(t_log, vel_log, label='Measured Velocity')
ax.xaxis.set_label_text('Time [s]')
ax.yaxis.set_label_text('Velocity [turns/s]')
ax.grid()
plt.legend()
plt.show()
"""