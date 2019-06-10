"""
Plot of final fuel consumption for different t_f settings.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import progressbar

from csm import ApolloCSM
import tools

matplotlib.rc('font',**{'family':'serif','size':14})
matplotlib.rc('text', usetex=True)

csm = ApolloCSM()

t_f = [150,250,350,450] # [s] Rendezvous duration

data = {key:None for key in t_f}
for _tf in t_f:
    with open('data/tf_%d_tc_2.pkl'%(_tf),'rb') as f:
        data[_tf] = pickle.load(f)

idx = 0

u = [tools.get_values(data_tf,'solver_output','optimizer','u')[idx].T for
     _,data_tf in data.items()]

apollo_fuel = 49.985879 # [kg] Apollo Mission-G fuel consumption
fuel_used = np.empty(len(t_f))
for j in range(len(t_f)):
    t_fuel,fuel = csm.compute_fuel_used(u[j],t_f[j])
    fuel_used[j] = fuel[-1]

barwidth=25
fig = plt.figure(1,figsize=(6.4,4.8))
plt.clf()
ax = fig.add_subplot(111)
tools.log_grid(ax,'y',[0,2])
ax.grid('x')
ax.set_yscale('log')
ax.axhline(apollo_fuel,color='red',linestyle='--',linewidth=2)
ax.text((t_f[0]+t_f[-1])*0.5,apollo_fuel+5,'Apollo $\\approx%.2f$~kg'%
        (apollo_fuel),ha='center')
ax.bar(t_f,fuel_used,width=barwidth,zorder=99,color='black')
for j in range(len(t_f)):
    ax.text(t_f[j],fuel_used[j]+0.5,'$\\approx%.2f$~kg'%
            (fuel_used[j]),ha='center')
plt.xticks(t_f,t_f)
ax.set_xlim([t_f[0]-1.5*barwidth,t_f[-1]+1.5*barwidth])
ax.set_xlabel('Rendezvous duration $t_f$ [s]')
ax.set_ylabel('Total fuel consumption [kg]')
plt.tight_layout()
plt.show(block=False)

fig.savefig('./figures/cost_vs_tf.pdf',
            bbox_inches='tight',format='pdf',transparent=True)
