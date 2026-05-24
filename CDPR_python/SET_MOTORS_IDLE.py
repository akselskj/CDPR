import odrive as odrive
from odrive.enums import AXIS_STATE_IDLE

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
    axes[motor_id] = odrv.axis0

for i in range(4):
    axis = axes[i]
    axis.requested_state = AXIS_STATE_IDLE
    print(f"Motor ", i, " set to IDLE")