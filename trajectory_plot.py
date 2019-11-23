"""
Plot the state and input trajectory.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import pickle
import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import progressbar

from csm import ApolloCSM
import tools

matplotlib.rc('font',**{'family':'serif','size':14})
matplotlib.rc('text', usetex=True)

matplotlib.rc('font',**{'family':'serif','size':14})
matplotlib.rc('text', usetex=True)

with open('data/tf_150_tc_2.pkl','rb') as f:
    data = pickle.load(f)

csm = ApolloCSM()

t_f = tools.get_values(data,'solver_input','t_f')[0]
q_f = tools.get_values(data,'solver_input','xf')[0]['q']
t_pulse_max = tools.get_values(data,'solver_input','t_pulse_max')[0]
r_a = tools.get_values(data,'solver_input','r_app')[0]

idx = 0
t_sol = tools.get_values(data,'solver_output','optimizer','t')[idx]
p_sol = tools.get_values(data,'solver_output','optimizer','p')[idx]
v_sol = tools.get_values(data,'solver_output','optimizer','v')[idx]
q_sol = tools.get_values(data,'solver_output','optimizer','q')[idx]
omega_sol = tools.get_values(data,'solver_output','optimizer','w')[idx]
u_sol = tools.get_values(data,'solver_output','optimizer','u')[idx]
vc_sol = tools.get_values(data,'solver_output','history','vc')[idx][-1]
t_nl = tools.get_values(data,'solver_output','nl_prop','t')[idx]
p_nl = tools.get_values(data,'solver_output','nl_prop','p')[idx]
v_nl = tools.get_values(data,'solver_output','nl_prop','v')[idx]
q_nl = tools.get_values(data,'solver_output','nl_prop','q')[idx]
omega_nl = tools.get_values(data,'solver_output','nl_prop','w')[idx]

ang_err_sol = np.array([np.rad2deg(2.*np.arccos(max(-1,min(1,tools.qmult(
    tools.qconj(q_sol_k),q_f)[0])))) for q_sol_k in q_sol.T])
ang_err_nl = np.array([np.rad2deg(2.*np.arccos(max(-1,min(1,tools.qmult(
    tools.qconj(q_nl_k),q_f)[0])))) for q_nl_k in q_nl.T])
vc_sol_norm1 = np.array([la.norm(vc_sol[k],ord=1)
                         for k in range(vc_sol.shape[0])])

rpy_sol = np.column_stack([tools.q2rpy(q_sol.T[k])
                           for k in range(q_sol.shape[1])])
rpy_nl = np.column_stack([tools.q2rpy(q_nl.T[k])
                          for k in range(q_nl.shape[1])])

time_app = np.array([t_nl[k] for k in range(p_nl.shape[1])
                     if la.norm(p_nl[:,k]-p_nl[:,-1])<r_a])

# translation
fig = plt.figure(1)
plt.clf()
ax = fig.add_subplot(211)
ax.grid()
ax.plot(t_sol,p_sol[0],color='red',linestyle='none',marker='.',markersize=5)
ax.plot(t_sol,p_sol[1],color='green',linestyle='none',marker='.',markersize=5)
ax.plot(t_sol,p_sol[2],color='blue',linestyle='none',marker='.',markersize=5)
ax.plot(t_nl,p_nl[0],color='red')
ax.plot(t_nl,p_nl[1],color='green')
ax.plot(t_nl,p_nl[2],color='blue')
ax.set_xlabel('Time $t$ [s]')
ax.set_ylabel('Position [m]')
ax.autoscale(tight=True)
y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.fill_between(time_app,
                np.repeat(y_lim[0],time_app.size),
                np.repeat(y_lim[1],time_app.size),
                linewidth=0,color='black',alpha=0.2,zorder=1)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax = fig.add_subplot(212)
ax.grid()
ax.plot(t_sol,v_sol[0],color='red',linestyle='none',marker='.',markersize=5)
ax.plot(t_sol,v_sol[1],color='green',linestyle='none',marker='.',markersize=5)
ax.plot(t_sol,v_sol[2],color='blue',linestyle='none',marker='.',markersize=5)
ax.plot(t_nl,v_nl[0],color='red')
ax.plot(t_nl,v_nl[1],color='green')
ax.plot(t_nl,v_nl[2],color='blue')
ax.set_xlabel('Time $t$ [s]')
ax.set_ylabel('Velocity [m/s]')
ax.autoscale(tight=True)
plt.tight_layout()
y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.fill_between(time_app,
                np.repeat(y_lim[0],time_app.size),
                np.repeat(y_lim[1],time_app.size),
                linewidth=0,color='black',alpha=0.2,zorder=1)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.show(block=False)

fig.savefig('./figures/translation.pdf',
            bbox_inches='tight',format='pdf',transparent=True)

# attitude
fig = plt.figure(2)
plt.clf()
ax = fig.add_subplot(211)
ax.grid()
ax.plot(t_sol,np.unwrap(rpy_sol[0],discont=359.)-360.,color='red',linestyle='none',marker='.',markersize=5)
ax.plot(t_sol,rpy_sol[1],color='green',linestyle='none',marker='.',markersize=5)
ax.plot(t_sol,rpy_sol[2],color='blue',linestyle='none',marker='.',markersize=5)
ax.plot(t_nl,np.unwrap(rpy_nl[0],discont=359.)-360.,color='red')
ax.plot(t_nl,rpy_nl[1],color='green')
ax.plot(t_nl,rpy_nl[2],color='blue')
ax.set_xlabel('Time $t$ [s]')
ax.set_ylabel('Roll, pitch, yaw [$^\circ$]')
ax.autoscale(tight=True)
y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.fill_between(time_app,
                np.repeat(y_lim[0],time_app.size),
                np.repeat(y_lim[1],time_app.size),
                linewidth=0,color='black',alpha=0.2,zorder=1)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
ax = fig.add_subplot(212)
ax.grid()
ax.plot(t_sol,np.rad2deg(omega_sol[0]),color='red',linestyle='none',marker='.',markersize=5)
ax.plot(t_sol,np.rad2deg(omega_sol[1]),color='green',linestyle='none',marker='.',markersize=5)
ax.plot(t_sol,np.rad2deg(omega_sol[2]),color='blue',linestyle='none',marker='.',markersize=5)
ax.plot(t_nl,np.rad2deg(omega_nl[0]),color='red')
ax.plot(t_nl,np.rad2deg(omega_nl[1]),color='green')
ax.plot(t_nl,np.rad2deg(omega_nl[2]),color='blue')
ax.set_xlabel('Time $t$ [s]')
ax.set_ylabel('Angular velocity [$^\circ$/s]')
ax.autoscale(tight=True)
plt.tight_layout()
y_lim = ax.get_ylim()
x_lim = ax.get_xlim()
ax.fill_between(time_app,
                np.repeat(y_lim[0],time_app.size),
                np.repeat(y_lim[1],time_app.size),
                linewidth=0,color='black',alpha=0.2,zorder=1)
ax.set_xlim(x_lim)
ax.set_ylim(y_lim)
plt.show(block=False)

fig.savefig('./figures/attitude.pdf',
            bbox_inches='tight',format='pdf',transparent=True)

# RCS thrusts
fig = plt.figure(3,figsize=(9.6,9.86))
plt.clf()
for i in range(4): # quad number
    for j in range(4): # thruster of quad
        k = i*4+j
        ax = fig.add_subplot(4,4,k+1)
        ax.grid()
        ax.plot(t_sol[:-1],u_sol[k]*1e3,color='black',marker='.',
                linestyle='none',markersize=5)
        ax.axhline(y=csm.t_pulse_max*1e3,color='red',linestyle='--',linewidth=1)
        ax.axhline(y=csm.t_pulse_min*1e3,color='blue',linestyle='--',
                   linewidth=1)
        ax.set_xlabel('Time $t$ [s]')
        ax.set_ylabel('Pulse width $\Delta t_k^{%d}$ [ms]'%(k+1))
        ax.set_title('Thruster $i=%d$'%(k+1))
        ax.autoscale(tight=True)
        yticks = (np.ceil(np.linspace(0,0.6,7)*1e3)).astype(int)
        xticks = (np.linspace(0,t_f,3)).astype(int)
        plt.yticks(yticks,yticks)
        plt.xticks(xticks,xticks)
        ax.set_ylim([0,t_pulse_max*1.1*1e3])
        if 'p_f' in csm.i2thruster[4*i+j]:
            y_lim = ax.get_ylim()
            x_lim = ax.get_xlim()
            ax.fill_between(time_app,
                            np.repeat(y_lim[0],time_app.size),
                            np.repeat(y_lim[1],time_app.size),
                            linewidth=0,color='black',alpha=0.2,zorder=1)
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
        
plt.tight_layout()
plt.show(block=False)

fig.savefig('./figures/inputs.pdf',
            bbox_inches='tight',format='pdf',transparent=True)
