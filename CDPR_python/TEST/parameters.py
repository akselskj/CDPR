import numpy as np

# -----------------------------
# Geometry & Parameters
# -----------------------------
l_p = 0.15        # platform width [m]
h_p = 0.02        # platform height [m]
L_f = 1.4         # frame width [m]
H_f = 0.8         # frame height [m]

mp  = 0.25        # mass [kg]
Izz = 8e-4        # inertia [kg·m²]
g   = 9.81        # gravity [m/s²]

f_min = 10.0      # min tension [N]
f_max = 80.0      # max tension [N]
f_c   = 45.0      # force-controlled cable tension [N]

r_d   = 20e-3     # drum radius [m]

# -----------------------------
# Mass matrix
# -----------------------------
M = np.diag([mp, mp, Izz])
Minv = np.diag([1/mp, 1/mp, 1/Izz])

# External wrench (gravity)
w_ext = np.array([0.0, -mp * g, 0.0])

# -----------------------------
# Frame and platform anchor points (2 x 4)
# -----------------------------
a = np.array([
    [-L_f/2, -L_f/2,  L_f/2,  L_f/2],
    [-H_f/2,  H_f/2,  H_f/2, -H_f/2]
])

b = np.array([
    [-l_p/6, -l_p/2,  l_p/2,  l_p/6],
    [-h_p/2,  h_p/2,  h_p/2, -h_p/2]
])

# -----------------------------
# Desired trajectory (circle)
# -----------------------------
R = 0.2           # radius [m]
omega = 1.0       # angular speed [rad/s]

# -----------------------------
# Hybrid control setup
# -----------------------------
n = 4             # number of cables
nd = 3            # platform DoFs
mu = n - nd       # redundancy

# Initial force-controlled cable
j_force = 3       # Python index (MATLAB j_force = 4)
j_prev = j_force

# Initial position-controlled cables
j_pos = [i for i in range(n) if i != j_force]

gamma = 0.995     # switching tolerance

# -----------------------------
# Motor parameters
# -----------------------------

motor_signs = [1, -1, 1, -1]

