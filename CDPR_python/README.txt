Software Documentation

This file describes the operation of the NTNU cable-driven parallel robot software developed as 
part of this project. The purpose of the document is to provide future users with sufficient 
information to safely operate, and modify the system. 

The software was developed and tested on Ubuntu 24.04.4 LTS using Python 3.12.3.

---- Software Structure ----

Brief description of each file:

- main.py           -- Main entry point. 
- control_loop.py   -- Outer control loop implementation for controller analysis. 
- camera.py         -- Vision system and ball tracking. 
- ping_pong_bot.py  -- Ball manipulation state machine and control loop. 
- motor_actions.py  -- ODrive communication and assorted motor commands. 
- geometry.py       -- Kinematic calculations, Jacobians and trajectory generation 
- utils.py          -- Utility functions. 
- parameters.py     -- System parameters and controller settings.
- SET_MOTORS_IDLE.py-- Connects to motors and sets them idle
- ball_plot.py, bounce_plot.py and plotting.py -- plotting scripts for balancing, bouncing and no ball


---- Startup Procedure ----

When the robot is to be used, follow this procedure:

1: Power the robot by plugging it to power and flipping the extension cord switch.
2: Connect Odrive and camera USB to computer
3: Start the software (main).
4: Perform the homing procedure (explained later).
5: Select the desired operating mode.
6: Experiment!


When running Main a rudimentary GUI will be presented in the terminal. 
At the time of writing this menu looks like this:

=== CDPR Main Menu === 
1 : Initialize / tighten cables 
2 : Run position control loop
3 : Run hybrid control loop
4 : Run Ping Pong demo
5 : Run velocity control loop
6 : HARD STOP (IDLE motors)
7 : move to home
8 : Move to user-defined pose
9 : Taut-cable pose observer (torque + live DK)
10 : Clear ODrive errors
11 : multithread position loop
12 : Run user control mode
13 : Run ball balancing
exit : Exit program


The different operating modes can be selected by entering the corresponding number 
in the terminal. Before using any operating mode, the robot should be homed to establish 
a known reference pose and tension the cables.

---- Homing Procedure ----

At startup, the robot assumes that the current motor positions correspond to the 
calibrated home position. If the platform is not physically located at home, the geometric 
calculations will be incorrect. In addition, the cables are often slightly slack after 
power-up and therefore need to be tensioned before operation.

The homing procedure places all motors in torque control mode and applies tension to the cables.

WARNING: Ensure that no objects or hands are inside the workspace before initiating the 
homing procedure. Any significant cable slack should be removed manually by winding the cable 
drums before tension is applied.

When prompted, enter y to tension the cables. Once tension has been established, verify that 
the cables are seated correctly on the pulleys and drums. If a cable has become slack during 
operation, it may slip off a guiding pulley or loosen on the drum. Reposition the cable if 
necessary. In some cases, sections of cable on the drum may remain loose even after tension 
has been applied. Lightly pushing the cable in the direction of the winding can help the cable 
settle properly onto the drum.

Next, manually align the platform such that the hole in the platform coincides with the hole 
in the backplate, and verify that the platform is level. With the Plexiglass enclosure installed, 
this alignment is most easily performed by pulling the two lower cables.


When the platform is correctly aligned, enter y to complete the homing procedure and return to 
the main menu. The robot is then ready for operation.


---- Operating Modes ----

Most operating modes will run continuously until interrupted by the user. The primary exceptions 
are Move to home and Move to user-defined pose, which terminate automatically once the target 
position has been reached.

To stop a running mode, press Ctrl+C. This causes the controller to exit its execution loop, save 
any collected data, and return to the main menu. The different operating modes are explained in 
the following. 

1: initialize / tighten cables
This mode allows for homing and rehoming the motor calibration as explained above. Saves new 
theta_home values for the motors.

2: Run position control loop
Runs the position controller used for the controller-comparison experiments presented in this thesis. 
The user may select from several predefined trajectories and adjust trajectory parameters such as 
size and speed in the function get_q_des in geometry.py.

This mode is primarily intended for controller evaluation and data collection. The implementation 
performs extensive logging and pulls more data than needed from the motor controllers. It is 
therefore not recommended as the basis for future real-time applications.

3: Run hybrid control loop
Runs the hybrid force-position controller used for the controller-comparison experiments presented 
in this thesis on the same trajectory as above.

4: Run Ping Pong Demo
Runs the integrated ball-manipulation demo including balancing, throwing, and repeated bouncing.

The mode begins by generating the homography used for transforming image coordinates into workspace 
coordinates. Ensure that all four ArUco markers are visible in the camera frame and that the camera 
is level before pressing y. The system captures two seconds of footage in order to generate a stable 
homography estimate.

Before enabling the controller, verify that the ball is visible in the camera image.

WARNING: If no ball is visible, other objects in the camera frame may be incorrectly identified as 
the ball, potentially causing aggressive robot motion.

The behavior of this demo and the ball balancing can be tuned in ping_pong_bot.py. For future 
real-time applications, the class-based structure used in this file is recommended over the 
function-oriented structure used in control_loop.py.

5: Run velocity control loop
Runs an experimental velocity controller. The implementation was never completed and is not 
recommended for normal use. However, users interested in further controller development are 
encouraged to experiment with and extend this mode.

6: HARD STOP (IDLE motors)
Immediately places all motors in the IDLE state. This mode serves as a backup stop in the event 
that the system unexpectedly returns to the main menu with the motors still in an active control mode.

7: Move to home
Moves the platform to the home position. This is useful for recentering the platform after 
experiments and before performing a new homing procedure.

8: Move to user-defined pose
Moves the platform along a smooth trajectory to a user-specified pose. This mode was primarily 
used for testing the kinematic implementation.

9: Taut-cable pose observer
Tensions the cables and allows the user to manually move the platform throughout the workspace. 
A live estimate of the platform pose based on direct kinematics is displayed on screen.

10: Clear ODrive errors
Clears all active ODrive errors. Note that this mode does not currently re-enable the brake 
resistors. If an error has disabled the brake resistors, the main program should be restarted 
after the error is cleared.

11: Multithread position loop
Runs an experimental multithreaded version of the position controller. The mode was developed 
to investigate whether separating communication and control calculations could increase the 
control-loop frequency. As the system is primarily limited by USB communication bandwidth, 
the performance improvement was negligible and development was discontinued.

12: Run user control mode
Allows the user to control the platform position using either the keyboard or the mouse.

13: Run ball balancing
Runs the ball-balancing controller while following the trajectory specified in get_q_des. 
The startup procedure is identical to that of the ping-pong demo.

exit: Exit program
By entering exit in the main menu the program will terminate.

Plotting Scripts
The plotting scripts are used to visualize data collected during experiments. To plot a dataset, 
provide the path to the desired log file in the function call at the bottom of the script. 
The displayed time interval can be adjusted using the T_START and T_WINDOW parameters near the 
top of the file.



I hope this codebase provides a useful foundation for future development of the CDPR platform. 
While this appendix describes the most important aspects of the system, the best understanding 
is gained through experimentation and exploration of the code. 

Have fun!