from motor_actions import *
from control_loop import *
from mock_motors import *
#import ping_pong_bot as ping
import ping_test as ping
import parameters as p
import gc
gc.disable()
import psutil, os
pr = psutil.Process(os.getpid())
pr.nice(0)



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


def main():
    motors, odrvs = discover_motors()
    print("CDPR control started")
    d_ref = geom.inverse_kinematics([0,0,0], p.a, p.b)
    d_home = geom.inverse_kinematics(p.home, p.a, p.b)
    delta_d = d_ref - d_home
    phi0 = []
    for i, axis in motors.items():
        phi0.append(axis.pos_estimate - delta_d[i]*p.motor_signs[i]/(2*np.pi*p.r_d))
    

    Kt = torque_constant(motors)

    while True:

        print_menu()
        user_input = input("Select option: ").strip()

        if user_input == "exit":
            print("Exiting program...")
            hard_stop(motors)
            break

        elif user_input.isdigit():
            cmd = int(user_input)

            if cmd == 1:
                phi0 = init_tension(motors, 0.2)
                print_motor_positions(motors)
                print("phi0 = ", phi0)

            elif cmd == 2:
                run_position_control_loop(odrvs, motors, phi0, Kt)

            elif cmd == 3:
                run_hybrid_control_loop(odrvs, motors, phi0, Kt)

            elif cmd == 4:
                ping.ping_pong_bot(odrvs, motors, phi0)

            elif cmd == 5:
                # this is not working correctly
                run_velocity_control_loop(odrvs, motors, phi0, Kt)

            elif cmd == 6:
                hard_stop(motors)

            elif cmd == 7:
                smooth_move_to_pose(motors, phi0, p.home)
                time.sleep(0.5)
                hard_stop(motors)
                                                
            elif cmd == 8:
                user_move_to_pose(motors, phi0)
                time.sleep(0.5)
                hard_stop(motors)
            
            elif cmd == 9:
                taut_observer_mode(motors, phi0, 0.2)

            elif cmd == 10:
                clear_all_errors(odrvs)

            elif cmd == 11:
                # this is not nesecary currently as the USB comunication is the bottleneck, not code runtime.
                # could be relevant if com channel is changed
                run_multithred_control(odrvs, motors, phi0, Kt)
            
            elif cmd == 12:
                run_keyboard_control_loop(odrvs, motors, phi0, Kt)
            
            elif cmd == 13:
                ping.balancing_bot(odrvs, motors, phi0)

            elif cmd == 14:
                ping.throw_ball(motors, phi0)

            elif cmd == 15:
                # This mode allows you to live tune the balancing PD, 
                # the other balancing mode is better when the tune is done
                ping.ball_balancing(odrvs, motors, phi0)    

            else:
                print("Unknown command")

        else:
            print("Invalid input")


main()