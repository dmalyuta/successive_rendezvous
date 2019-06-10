"""
Simulate the Apollo CSM dynamics.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np
import numpy.linalg as la
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

import csm

# Initial state
# position_init = np.array([0.,0.,0.])
# velocity_init = np.array([0.,0.,0.])
# quaternion_init = np.array([1.,0.,0.,0.])
# ang_vel_init = np.deg2rad(np.array([0.,0.,0.]))
position_init = np.random.randn(3)
velocity_init = np.random.randn(3)
quaternion_init = np.random.randn(4)
quaternion_init /= la.norm(quaternion_init)
ang_vel_init = np.deg2rad(np.random.randn(3))
state_init = np.concatenate([position_init,velocity_init,
                             quaternion_init,ang_vel_init])

# Constant input
n_thrusters = 16
u_const = np.zeros(n_thrusters)
lbf2N = lambda x: x*4.448222 # lbf to N
thrust_max = lbf2N(100.)
u_const = thrust_max*np.random.uniform(0.,1.,size=(n_thrusters))

# Dynamics
csm_nl = csm.ApolloCSM(dynamics='nonlinear')
csm_l = csm.ApolloCSM(dynamics='linear',ref=dict(x=state_init,u=u_const))

# Control input
def rcs_input(t,x):
    return u_const

# Simulation
T_sim = 10. # [s] Simulation duration
max_step = 1.#1e-2 # [s] Coarsest time step
simout_nl = solve_ivp(lambda t,x: csm_nl.dxdt(x,rcs_input(t,x)),
                      t_span=(0,T_sim),y0=state_init,max_step=1e-2)
simout_l = solve_ivp(lambda t,x: csm_l.dxdt(x,rcs_input(t,x)),
                     t_span=(0,T_sim),y0=state_init,max_step=1e-2)

# Plot results
fig = plt.figure(1)
plt.clf()
# Position evolution
ax = fig.add_subplot(411)
ax.plot(simout_nl.t,simout_nl.y[0],color='red',label='$p_x$')
ax.plot(simout_nl.t,simout_nl.y[1],color='green',label='$p_y$')
ax.plot(simout_nl.t,simout_nl.y[2],color='blue',label='$p_z$')
ax.plot(simout_l.t,simout_l.y[0],color='red',label='$p_x$',linestyle='--')
ax.plot(simout_l.t,simout_l.y[1],color='green',label='$p_y$',linestyle='--')
ax.plot(simout_l.t,simout_l.y[2],color='blue',label='$p_z$',linestyle='--')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Position [m]')
ax.legend()
# Velocity evolution
ax = fig.add_subplot(412)
ax.plot(simout_nl.t,simout_nl.y[3],color='red',label='$v_x$')
ax.plot(simout_nl.t,simout_nl.y[4],color='green',label='$v_y$')
ax.plot(simout_nl.t,simout_nl.y[5],color='blue',label='$v_z$')
ax.plot(simout_l.t,simout_l.y[3],color='red',label='$v_x$',linestyle='--')
ax.plot(simout_l.t,simout_l.y[4],color='green',label='$v_y$',linestyle='--')
ax.plot(simout_l.t,simout_l.y[5],color='blue',label='$v_z$',linestyle='--')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Velocity [m/s]')
ax.legend()
# Orientation evolution
ax = fig.add_subplot(413)
ax.plot(simout_nl.t,simout_nl.y[6],color='black',label='$q_w$')
ax.plot(simout_nl.t,simout_nl.y[7],color='red',label='$q_x$')
ax.plot(simout_nl.t,simout_nl.y[8],color='green',label='$q_y$')
ax.plot(simout_nl.t,simout_nl.y[9],color='blue',label='$q_z$')
ax.plot(simout_l.t,simout_l.y[6],color='black',label='$q_w$',linestyle='--')
ax.plot(simout_l.t,simout_l.y[7],color='red',label='$q_x$',linestyle='--')
ax.plot(simout_l.t,simout_l.y[8],color='green',label='$q_y$',linestyle='--')
ax.plot(simout_l.t,simout_l.y[9],color='blue',label='$q_z$',linestyle='--')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Quaternion component')
ax.legend()
# Angular velocity evolution
ax = fig.add_subplot(414)
ax.plot(simout_nl.t,np.rad2deg(simout_nl.y[10]),color='red',label='$\omega_x$')
ax.plot(simout_nl.t,np.rad2deg(simout_nl.y[11]),color='green',label='$\omega_y$')
ax.plot(simout_nl.t,np.rad2deg(simout_nl.y[12]),color='blue',label='$\omega_z$')
ax.plot(simout_l.t,np.rad2deg(simout_l.y[10]),color='red',label='$\omega_x$',linestyle='--')
ax.plot(simout_l.t,np.rad2deg(simout_l.y[11]),color='green',label='$\omega_y$',linestyle='--')
ax.plot(simout_l.t,np.rad2deg(simout_l.y[12]),color='blue',label='$\omega_z$',linestyle='--')
ax.set_xlabel('Time [s]')
ax.set_ylabel('Angular velocity [deg/s]')
ax.legend()
plt.tight_layout()
plt.show(block=False)
