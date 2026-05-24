import odrive as odrive
import time
from odrive.enums import *

SERIAL_TO_MOTOR = {
    "394A353B3231": 0,
    "393E353C3231": 1,
    "393D35443231": 2,
    "394A35513231": 3,
}

axes = {}

print("Connecting to ODrives by serial number...")

for serial, motor_id in SERIAL_TO_MOTOR.items():
    print(f"Connecting to motor {motor_id} (serial {serial})")
    odrv = odrive.find_any(serial_number=serial, timeout=20)
    axis = odrv.axis0
    axis.requested_state = AXIS_STATE_IDLE
    print(f"Motor {motor_id} set to IDLE")
    axes[motor_id] = axis

time.sleep(0.5)

print("\nReading motor positions (Ctrl+C to stop)")

try:
    while True:
        line = []
        for i, axis in axes.items():
            pos = axis.pos_estimate
            vel = axis.vel_estimate
            line.append(f"M{i}: {pos: .4f}")
        print(" | ".join(line), end="\r")
        time.sleep(0.05)

except KeyboardInterrupt:
    print("\nStopped.")
