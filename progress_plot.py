"""
Plot of successive convexification algorithm progress.

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

with open('data/tf_150_tc_2.pkl','rb') as f:
    data = pickle.load(f)

csm = ApolloCSM()

idx = 0

t_f = tools.get_values(data,'solver_input','t_f')[idx]
w_tr = tools.get_values(data,'solver_input','w_tr')[idx]
p_e_max = tools.get_values(data,'solver_input','ppgt_tol')[idx]['p']
v_e_max = tools.get_values(data,'solver_input','ppgt_tol')[idx]['v']
w_e_max = tools.get_values(data,'solver_input','ppgt_tol')[idx]['w']
ang_e_max = tools.get_values(data,'solver_input','ppgt_tol')[idx]['ang']

u = tools.get_values(data,'solver_output','history','u')[idx]
u_opt = tools.get_values(data,'solver_output','optimizer','u')[idx].T
J = tools.get_values(data,'solver_output','history','cost')[idx]
J_tr = tools.get_values(data,'solver_output','history','J_tr')[idx]
x_ppgt_tf = np.array([
    z[-1] for z in tools.get_values(
        data,'solver_output','history','x_ppgt')[idx][1:]])
x_opt_tf = np.array([
    z[-1] for z in tools.get_values(
        data,'solver_output','history','x')[idx][1:]])

iters = list(range(1,len(J)))
j_max = iters[-1]
tr = np.array(J_tr[1:])/w_tr
fuel_used = np.empty(j_max)
for j in progressbar.progressbar(range(j_max)):
    t_fuel,fuel = csm.compute_fuel_used(u[j],t_f)
    fuel_used[j] = fuel[-1]
apollo_fuel = 49.985879 # [kg] Apollo Mission-G fuel consumption
p_e = np.array([la.norm(x_ppgt_tf[k,:3]-x_opt_tf[k,:3],ord=np.inf)
                for k in range(j_max)])
v_e = np.array([la.norm(x_ppgt_tf[k,3:6]-x_opt_tf[k,3:6],ord=np.inf)
                for k in range(j_max)])
w_e = np.array([np.rad2deg(la.norm(x_ppgt_tf[k,10:]-x_opt_tf[k,10:],ord=np.inf))
                for k in range(j_max)])
ang_e = np.array([np.rad2deg(2*np.arccos(tools.qmult(tools.qconj(
    x_opt_tf[k,6:10]),x_ppgt_tf[k,6:10])[0])) for k in range(j_max)])

fig = plt.figure(1,figsize=(9,9))
plt.clf()

ax = fig.add_subplot(221)
tools.log_grid(ax,'x',[0,2])
tools.log_grid(ax,'y',[0,3])
ax.axhline(apollo_fuel,color='red',linestyle='--',linewidth=2)
ax.text(j_max-1,apollo_fuel+5,'Apollo $\\approx%.2f$~kg'%(apollo_fuel),ha='right')
ax.text(j_max-1,fuel_used[-1]-0.5,'This work $\\approx%.2f$~kg'%(fuel_used[-1]),
        ha='right',va='top')
ax.loglog(iters,fuel_used,color='black')
ax.set_xlabel('Algorithm iteration $j$')
ax.set_ylabel('Total fuel consumption [kg]')
ax.set_ylim([1e0,1e3])
ax.set_xlim([1,j_max])

ax = fig.add_subplot(223)
tools.log_grid(ax,'x',[0,2])
tools.log_grid(ax,'y',[-10,1])
ax.loglog(iters,tr,color='black')
ax.set_xlabel('Algorithm iteration $j$')
ax.set_ylabel('Trust region size $\sum_{k=1}^{N-1}\hat\eta_k$')
ax.set_ylim([1e-6,10])
ax.set_xlim([1,j_max])

ax = fig.add_subplot(422)
tools.log_grid(ax,'x',[0,2])
tools.log_grid(ax,'y',[-6,2])
ax.axhline(p_e_max,color='red',linestyle='--',linewidth=2)
ax.text(j_max-1,p_e_max+5e-3,'$p_{e,\max}=%d$~cm'%(p_e_max*1e2),ha='right')
ax.loglog(iters,p_e,color='black')
ax.set_xlabel('Algorithm iteration $j$')
ax.set_ylabel('$p_e$ [m]')
ax.set_xlim(1,j_max)
ax.set_ylim([1e-4,3])

ax = fig.add_subplot(424)
tools.log_grid(ax,'x',[0,2])
tools.log_grid(ax,'y',[-6,2])
ax.axhline(v_e_max,color='red',linestyle='--',linewidth=2)
ax.text(j_max-1,v_e_max+5e-4,'$v_{e,\max}=%d$~mm/s'%(v_e_max*1e3),ha='right')
ax.loglog(iters,v_e,color='black')
ax.set_xlabel('Algorithm iteration $j$')
ax.set_ylabel('$v_e$ [m/s]')
ax.set_xlim(1,j_max)
ax.set_ylim([1e-6,3e-1])

ax = fig.add_subplot(426)
tools.log_grid(ax,'x',[0,2])
tools.log_grid(ax,'y',[-4,2])
ax.axhline(ang_e_max,color='red',linestyle='--',linewidth=2)
ax.text(j_max-1,ang_e_max+0.1,'$\\theta_{e,\max}=%.1f^\circ$'%(ang_e_max),
        ha='right')
ax.loglog(iters,ang_e,color='black')
ax.set_xlabel('Algorithm iteration $j$')
ax.set_ylabel('$\\theta_e$ [$^\circ$]')
ax.set_xlim(1,j_max)
ax.set_ylim([1e-2,5e0])

ax = fig.add_subplot(428)
tools.log_grid(ax,'x',[0,2])
tools.log_grid(ax,'y',[-8,1])
ax.axhline(w_e_max,color='red',linestyle='--',linewidth=2)
ax.text(j_max-1,w_e_max+3e-3,'$\omega_{e,\max}=%.2f$ $^\circ$/s'%(w_e_max),ha='right')
ax.loglog(iters,w_e,color='black')
ax.set_xlabel('Algorithm iteration $j$')
ax.set_ylabel('$\omega_e$ [$^\circ$/s]')
ax.set_xlim(1,j_max)
ax.set_ylim([1e-5,5e-2])

plt.tight_layout()
plt.show(block=False)

fig.savefig('./figures/progress.pdf',
            bbox_inches='tight',format='pdf',transparent=True)
