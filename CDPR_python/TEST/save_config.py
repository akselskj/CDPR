import odrive
import time

SERIAL_TO_MOTOR = {
    "394A353B3231": 0,
    "393E353C3231": 1,
    "393D35443231": 2,
    "394A35513231": 3,
}
for serial, motor_id in SERIAL_TO_MOTOR.items():
    print(f"Connecting to motor {motor_id} ({serial})")


    odrv = odrive.find_any(serial_number=serial, timeout=10)
    print(f"Saving configuration for motor {motor_id}")
    axis = odrv.axis0

    odrv.config.dc_bus_overvoltage_trip_level = 36
    odrv.config.brake_resistor0.resistance = 2.0
    odrv.config.brake_resistor0.enable = True
    odrv.config.brake_resistor0.enable_dc_bus_voltage_feedback = True
    odrv.config.brake_resistor0.dc_bus_voltage_feedback_ramp_end = 30
    odrv.clear_errors()

    axis.controller.config.input_filter_bandwidth = 25.0

    axis.controller.config.pos_gain = 30.0     
    axis.controller.config.vel_gain = 0.5
    axis.controller.config.vel_integrator_gain = 0.5
    
    axis.controller.config.vel_limit = 5.0
    axis.controller.config.vel_limit_tolerance = 1.2
    
    try:
        odrv.save_configuration()
    except Exception as e:
        print(f"Motor {motor_id}: disconnected after save (EXPECTED)")
    
    time.sleep(2.0)
    print(f"Motor {motor_id} done")
        

    
