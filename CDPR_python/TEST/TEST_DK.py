import numpy as np
import matplotlib.pyplot as plt
import parameters as p
import utils
import geometry as geom


# --- Desired circular trajectory ---
N = 300
t = np.linspace(0, 2*np.pi, N)

q_des = np.zeros((N, 3))
q_des[:, 0] = p.R * np.cos(t)
q_des[:, 1] = -p.R * np.sin(t)   # start at bottom, clockwise
q_des[:, 2] = 0.0


# --- Run IK → DK ---
q_est = np.zeros_like(q_des)
q_prev = q_des[0].copy()

for k in range(N):
    l = geom.inverse_kinematics(q_des[k], p.a, p.b)

    q_hat, _ = geom.direct_kinematics(
        l_act=l,
        v_cable=np.zeros(4),
        a=p.a,
        b=p.b,
        q_prev=q_prev
    )

    q_est[k] = q_hat
    q_prev = q_hat


# --- Plot ---
plt.figure()
plt.plot(q_des[:,0], q_des[:,1], 'k--', label='desired (IK)')
plt.plot(q_est[:,0], q_est[:,1], 'r', label='estimated (DK)')
plt.axis('equal')
plt.legend()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('IK → DK consistency test')
plt.show()


