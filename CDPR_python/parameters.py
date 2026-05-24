import numpy as np


# -- Geometry & Parameters --
l_p = 0.150       # platform width [m]
h_p = 0.016       # platform height [m]
L_f = 1.45        # frame width [m]
H_f = 0.865        # frame height [m]

#workspace limits
WORKSPACE_X_MIN = -0.5
WORKSPACE_X_MAX = 0.5
WORKSPACE_Y_MIN = -0.4
WORKSPACE_Y_MAX = 0.4

f_min = 0.1      # min tension [N]
f_max = 1.0      # max tension [N]
f = 0.3
f_c   = [1.5*f, 0.1*f, 0.4*f, 1.5*f]      # force-controlled cable tension [N]

r_d   = 20e-3     # drum radius [m]


# -- Frame and platform anchor points (2 x 4) --
a = np.array([
    [-L_f/2, -L_f/2,  L_f/2,  L_f/2],
    [-H_f/2,  H_f/2,  H_f/2, -H_f/2]
])

b = np.array([
    [-l_p/6, -l_p/2,  l_p/2,  l_p/6],
    [-h_p/2,  h_p/2,  h_p/2, -h_p/2]
])


# -- home position --
home = [0, -0.02, 0]    # hole
#home = [0, -0.3, 0]      # low home, tape


# -- Desired trajectory parameters --
R = 0.15           # radius [m]
omega = 1.5          # angular speed [rad/s]
MAX_VEL = 3


# -- Hybrid control setup --

n = 4             # number of cables
nd = 3            # platform DoFs
mu = n - nd       # redundancy

# Initial force-controlled cable
j_force = 3       # index of cable under force control
j_prev = j_force  # previous force-controlled cable
j_pos = [i for i in range(n) if i != j_force]   # Initial position-controlled cables

gamma = 0.998     # switching tolerance


# -- Ballistic parameters --
g = 9.81          # gravity [m/s^2]
T_flight = 0.4    # flight time [s]


# -- Motor parameters --
motor_signs = [1, -1, 1, -1]

# -- Camera parameters --
cam_latency = 0.015
