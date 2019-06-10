"""
Apply successive convexification to solve the rendezvous problem.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import time
import pickle
import numpy as np
import numpy.linalg as la
import scipy.interpolate as siplt
import cvxpy as cvx

import csm as apollo_csm
import tools

def terminal_conditions(data):
    """
    Create the terminal state of the CSM.

    Parameters
    ----------
    data : dict
        Problem data.
    """
    csm = data['csm']
    
    q_csm_f = tools.qmult(data['lm']['q'],csm.q_dock)
    if q_csm_f.dot(data['x0']['q'])<0.:
        # This makes sure q_csm_f is the same as the destination quaternion for
        # SLERP, which will flip the sign of the destination quaternion in order
        # to take the shortest path
        q_csm_f *= -1.
    p_csm_f = (data['lm']['p']+
               tools.rotate(csm.r_dock_lm,data['lm']['q'])-
               tools.rotate(csm.r_dock_csm,q_csm_f))
    w_csm_f = tools.rotate(data['lm']['w'],tools.qconj(csm.q_dock))
    # L_v_dock: docking impact velocity in LM frame
    L_v_dock = data['v_dock']*tools.rotate(np.array([1.,0.,0.]),csm.q_dock)
    v_csm_f = tools.rotate(L_v_dock+np.cross(data['lm']['w'],csm.r_dock_lm),
                           data['lm']['q'])
    
    data['xf'] = dict(p=p_csm_f,v=v_csm_f,q=q_csm_f,w=w_csm_f)    

def initial_guess(data):
    """
    Compute the initial state and input trajectory guess.
    
      position: linear interpolation
      velocity: constant based on position
      quaternion: slerp
      angular rate: constant based on quaternion
      input: zero
    
    Parameters
    ----------
    data : dict
        Problem data.
    """
    N = data['N']
    t_f = data['t_f']
    x0 = data['x0']
    xf = data['xf']
    csm = data['csm']
    
    alpha = np.linspace(0.,1.,N+1)
    
    p_traj = np.outer(1.-alpha,x0['p'])+np.outer(alpha,xf['p'])
    v_traj = np.row_stack([(xf['p']-x0['p'])/t_f for _ in range(N+1)])
    q_traj,w = tools.slerp(x0['q'],xf['q'],np.linspace(0.,1.,N+1))
    w_traj = np.row_stack([w for _ in range(N+1)])/t_f

    data['state_traj_init'] = np.column_stack([p_traj,v_traj,q_traj,w_traj])
    data['input_traj_init'] = csm.t_pulse_min*np.ones((N,csm.M))

    for i in range(csm.M):
        if i in data['thrusters_off']:
            data['input_traj_init'][:,i] = 0.

def compile_lrtop(data):
    """
    Create the local rendezvous trajectory optimization problem (LRTOP).

    Parameters
    ----------
    data : dict
        Problem data.
    """
    csm = data['csm']
    N = data['N']
    p_traj = data['state_traj_init'][:,:3]
    v_traj = data['state_traj_init'][:,3:6]
    w_traj = data['state_traj_init'][:,10:13]
    
    state_init = np.concatenate([data['x0']['p'],data['x0']['v'],
                                 data['x0']['q'],data['x0']['w']])
    state_final = np.concatenate([data['xf']['p'],data['xf']['v'],
                                  data['xf']['q'],data['xf']['w']])
    
    # Compute scaling terms
    # state (=P_x*state_scaled+p_x)
    p_center = np.mean(p_traj,axis=0)
    v_center = np.mean(v_traj,axis=0)
    q_center = np.zeros(4)
    w_center = np.mean(w_traj,axis=0)
    p_box = tools.bounding_box(p_traj-p_center)
    v_box = tools.bounding_box(v_traj-v_center)
    q_box = np.ones(4)
    w_box = tools.bounding_box(w_traj-w_center)
    p_x = np.concatenate([p_center,v_center,q_center,w_center])
    P_x = np.diag(np.concatenate([p_box,v_box,q_box,w_box]))
    P_x_inv = la.inv(P_x)
    data['scale_x'] = lambda x: P_x_inv.dot(x-p_x)
    # virtual control (=P_v*vc_scaled)
    p_box = tools.bounding_box(p_traj)
    v_box = tools.bounding_box(v_traj)
    q_box = np.ones(4)
    w_box = tools.bounding_box(w_traj)
    P_v = np.diag(np.concatenate([p_box,v_box,q_box,w_box]))
    # input (=P_u*input_scaled+p_u)
    p_u = np.array([0.5*data['t_pulse_max'] for i in range(csm.M)])
    P_u = np.diag(p_u)
    # fuel (impulse) cost (=P_xi*xi_scaled+p_xi)
    # Heuristic: compute as having a quarter of the thrusters active at minimum
    # pulse width, the whole time
    mean_active_thruster_count = 0.25*csm.M
    mean_pulse_width = csm.t_pulse_min
    p_xi = 0.
    P_xi = mean_active_thruster_count*mean_pulse_width*N
    
    # General quantites
    data['t_grid'] = np.linspace(0.,data['t_f'],N+1) # discretization time grid
    n_x = data['state_traj_init'].shape[1]
    n_u = csm.M
    e = -tools.rotate(np.array([1.,0.,0.]),csm.q_dock) # dock axis in LM frame
    I_e = tools.rotate(e,data['lm']['q']) # dock axis in inertial frame
    data['dock_axis'] = I_e

    # Optimization variables
    # non-dimensionalized
    x_hat = [cvx.Variable(n_x) for k in range(N+1)]
    v_hat = [cvx.Variable(n_x) for k in range(N+1)]
    xi_hat = cvx.Variable(N+1)
    u_hat = [cvx.Variable(n_u) for k in range(N)]
    eta_hat = cvx.Variable(N) # quadratic trust region size
    # dimensionalized (physical units)
    x = [P_x*x_hat[k]+p_x for k in range(N+1)] # unscaled state
    xi = P_xi*xi_hat+p_xi
    u = [P_u*u_hat[k]+p_u for k in range(N)] # unscaled control
    v = [P_v*v_hat[k] for k in range(N)] # virtual control
    data['lrtop_var'] = dict(x=x,u=u,xi=xi,v=v)

    # Optimization parameters
    A = [cvx.Parameter((n_x,n_x)) for k in range(N)]
    B = [cvx.Parameter((n_x,n_u)) for k in range(N)]
    r = [cvx.Parameter(n_x) for k in range(N)]
    u_lb = [cvx.Parameter(n_u,value=np.zeros(n_u)) for k in range(N)]
    stc_lb = [cvx.Parameter(n_u) for k in range(N)]
    stc_q = cvx.Parameter(N+1)
    x_prev_hat = [cvx.Parameter(n_x) for k in range(N+1)]
    data['lrtop_par'] = dict(A=A,B=B,r=r,u_lb=u_lb,stc_lb=stc_lb,stc_q=stc_q,
                             x_prev_hat=x_prev_hat)
    
    # Cost
    # minimum-impulse
    J = xi_hat[-1]
    # trust region penalty
    J_tr = data['w_tr']*sum(eta_hat)
    # virtual control penalty
    J_vc = data['w_vc']*sum([cvx.norm1(v_hat[k]) for k in range(N)])
    data['lrtop_cost'] = dict(J=J,J_tr=J_tr,J_vc=J_vc)
    cost = cvx.Minimize(J+J_tr+J_vc)
    
    # constraints
    constraints = []
    constraints += [x[k+1]==A[k]*x[k]+B[k]*u[k]+r[k]+v[k] for k in range(N)]
    constraints += [xi[k+1]==xi[k]+sum(u[k]) for k in range(N)]
    constraints += [x[0]==state_init,x[-1]==state_final,xi[0]==0]
    constraints += [(x[k][:3]-data['lm']['p']).T*I_e>=
                    cvx.norm(x[k][:3]-data['lm']['p'])*
                    np.cos(np.deg2rad(data['gamma']))
                    for k in range(N+1)]
    constraints += [u[k]<=data['t_pulse_max'] for k in range(N)]
    constraints += [u[k]>=u_lb[k] for k in range(N)]
    constraints += [u[k][i] == 0. for k in range(N) for i in range(n_u)
                    if i in data['thrusters_off']]
    constraints += [cvx.quad_form(x_hat[k+1]-x_prev_hat[k+1],np.eye(n_x))
                    <=eta_hat[k] for k in range(N)]
    constraints += [stc_lb[k][i]*u[k][i]==0
                    for k in range(N) for i in range(n_u)]
    constraints += [stc_q[k]*u[k][i]==0 for k in range(N)
                    for i in range(n_u) if 'p_f' in csm.i2thruster[i]]
    constraints += [stc_q[k+1]*
                    (tools.rqpmat(data['xf']['q'])[0].dot(np.diag([1,-1,-1,-1]))
                     *x[k][6:10]-np.cos(0.5*np.deg2rad(data['ang_app'])))<=0
                    for k in range(N)]

    data['lrtop'] = cvx.Problem(cost,constraints)

def discretize(data,bar_x,bar_u):
    """
    Discretize the basic rendezvous trajectory optimization problem.

    Parameters
    ----------
    data : dict
        Problem data.
    bar_x : np.ndarray
        Array whose k-th row is the previous iteration's discrete-time state
        trajectory value at time step k.
    bar_u : np.ndarray
        Array whose k-th row is the previous iteration's discrete-time input
        trajectory value at time step k.

    Returns
    -------
    x_nl : np.ndarray
        Array whose k-th row is the non-linear propagated state trajectory with
        reinitialization at the start of each control interval.
    """
    csm = data['csm']
    eps_machine = np.finfo(float).eps
    A,B,r,x_nl = csm.dltv(data['t_grid'],bar_x,bar_u,
                          **data['ode_options']['dltv'])
    for k in range(data['N']):
        # Remove small elements
        _A,_B,_r = A[k],B[k],r[k]
        _A[np.abs(_A)<eps_machine] = 0.
        _B[np.abs(_B)<eps_machine] = 0.
        _r[np.abs(_r)<eps_machine] = 0.
        data['lrtop_par']['A'][k].value = _A
        data['lrtop_par']['B'][k].value = _B
        data['lrtop_par']['r'][k].value = _r
    return x_nl

def solve_lrtop(data,bar_x,bar_u):
    """
    Solve the local rendezvous trajectory optimization problem.

    Parameters
    ----------
    data : dict
        Problem data.
    bar_x : np.ndarray
        Array whose k-th row is the previous iteration's discrete-time state
        trajectory value at time step k.
    bar_u : np.ndarray
        Array whose k-th row is the previous iteration's discrete-time input
        trajectory value at time step k.

    Returns
    -------
    bar_x : np.ndarray
        New value for bar_x.
    bar_u : np.ndarray
        New value for bar_u.
    L : np.ndarray
        Achieved linearized cost.
    """
    csm = data['csm']
    N = data['N']
    # Update parameters
    for k in range(N):
        data['lrtop_par']['x_prev_hat'][k+1].value = data['scale_x'](bar_x[k+1])
        data['lrtop_par']['stc_lb'][k].value = (
            bar_u[k]-csm.t_pulse_min).clip(max=0)
    data['lrtop_par']['stc_q'].value = (
        la.norm(bar_x[:N+1,:3]-data['xf']['p'],axis=1)-data['r_app']).clip(max=0)
    # Call solver
    data['lrtop'].solve(**data['solver_options'])
    if data['lrtop'].status != cvx.OPTIMAL:
        if data['lrtop'].status == cvx.OPTIMAL_INACCURATE:
            print('(warning) inaccurate solution')
        else:
            raise RuntimeError('optimization failed (status: %s)'%
                               (data['lrtop'].status))

    # Extract solution
    bar_x = np.row_stack([data['lrtop_var']['x'][k].value for k in range(N+1)])
    bar_u = np.row_stack([data['lrtop_var']['u'][k].value for k in range(N)])
    L = data['lrtop_var']['xi'][-1].value

    return bar_x,bar_u,L

def solution_reset(data,history,j,bar_x,bar_u):
    """
    Resetting procedure to recover from STC locking feasibility issues.

    Parameters
    ----------
    data : dict
        Problem data.
    history : list
        Iteration history.
    j : int
        Current iteration number.
    bar_x : np.ndarray
        Array whose k-th row is the previous iteration's discrete-time state
        trajectory value at time step k.
    bar_u : np.ndarray
        Array whose k-th row is the previous iteration's discrete-time input
        trajectory value at time step k.

    Returns
    -------
    bar_x : np.ndarray
        New value for bar_x.
    bar_u : np.ndarray
        New value for bar_u.
    tbl_action : str
        Action taken.
    """
    csm = data['csm']
    u_lb = data['lrtop_par']['u_lb']
    
    eps_machine = np.finfo(float).eps
    ppgt_err_hist = np.array([not history[k]['ppgt_err'] for k in range(1,j)])
    lb_viol_hist = np.array([not history[k]['lb_viol'] for k in range(1,j)])
    good_iterates = np.argwhere(np.logical_and(ppgt_err_hist,lb_viol_hist))
    if good_iterates.size==0:
        good_iterates = np.argwhere(ppgt_err_hist)
    tbl_action = ''
    
    if good_iterates.size>0:
        # Select latest good iterate
        j_star = np.max(good_iterates)
        tbl_action = '%d'%(j_star+1)
        # Reset "previous iteration" trajectory to this iterate
        bar_x = history[j_star]['x']
        bar_u = history[j_star]['u']
        # Force all inputs that were \in (0,t_pulse_min), i.e.
        # violating the off-or-lower-bounded condition at the subsequent
        # iteration, to be >=t_pulse_min.
        for k in range(data['N']):
            u_lb[k].value = np.zeros(csm.M)
            for l in range(j_star+1,j):
                _u = history[l]['u']
                u_lb[k].value[np.logical_and(
                    _u[k]<=csm.t_pulse_min-eps_machine,
                    _u[k]>=eps_machine)] = csm.t_pulse_min
            bar_u[k] = bar_u[k].clip(min=u_lb[k].value)
    
    return bar_x,bar_u,tbl_action

def solve_rendezvous(csm,data):
    """
    Solve the spacecraft rendezvous problem.

    Parameters
    ----------
    csm : ApolloCSM
        Apollo Command and Service Module (CSM) object
    data : dict
        Dictionary of required data, with fields:
          - N: node count of discrete time grid
          - t_f: rendezvous duration [s]
          - t_pulse_max: maximum pulse width [s]
          - max_iter: maximum number of successive convexification iterations
          - w_tr: trust region size penalty weight
          - w_vc: virtual control penalty weight
          - ppgt_tol: threshold on terminal propagation error accuracy
              - p: position error [m]
              - v: velocity error [m/s]
              - ang: angular error [deg]
              - w: angular velocity error [deg/s]
          - dJ_tol: cost threshold at which to stop the iterations
          - use_stop_crit: if ``True``, use cost change as stopping criterion
          - v_dock: impact speed [m/s] of CSM/LM docking adapters at rendezvous
          - r_app: approach radius [m] for thrust plume impingement
          - ang_app: mass error [deg] wrt final quaternion below approach radius
          - gamma: approach cone half-angle [deg]
          - thrusters_off: a list of thrusters that are to be kept turned off
          - solver_options: options to pass to cvx.solve
          - ode_options: options to pass to scipy.integrate.solve_ivp

    Returns
    -------
    sol : dict
        Converged trajectory and algorithm runtime history.
    """
    original_data = data
    data = data.copy() # Make data local
    data['csm'] = csm
    
    # Parameters
    N = data['N']
    t_f = data['t_f']
    max_iter = data['max_iter']
    w_tr = data['w_tr']
    w_vc = data['w_vc']
    ppgt_tol = data['ppgt_tol']
    dJ_tol = data['dJ_tol']
    ode_options = data['ode_options']    
    eps_machine = np.finfo(float).eps

    # Compute terminal state
    terminal_conditions(data)
    original_data['xf'] = data['xf']

    # Compute initial trajectory guess
    initial_guess(data)
    state_init = np.concatenate([
        data['x0']['p'],data['x0']['v'],data['x0']['q'],data['x0']['w']])

    # Pre-compile the LRTOP
    compile_lrtop(data)
    
    # Successive convexification progress table
    tbl_col_size = [3,8,8,8,8,8,8,8,8,8,8]
    tbl_headers = ['i','J_tr','J_vc','J','u_viol','x_viol',
                   'p_err','v_err','ang_err','w_err','reset']
    print('successive convexification progress:\n')
    tools.print_table(tbl_headers,tbl_col_size)

    # Storage variable for iteration data
    iteration_data = [dict(
        # state trajectory output by optimizer
        x=None,
        # state trajectory propagated through non-linear dynamics using the
        # previous iteration's input, used for DLTV creation
        x_nl=None,
        # Propagation over [0,t_f] time grid
        t_ppgt=None,
        # Propagation over [0,t_f] states
        x_ppgt=None,
        # input trajectory output by optimizer
        u=None,
        # virtual control trajectory output by optimizer
        vc=None,
        # how many inputs violate the off-or-lower-bounded constraint
        lb_viol=None,
        # how many state constraints are violated
        x_viol=None,
        # original cost
        cost=None,
        # trust region
        J_tr=None,
        # virtual control
        J_vc=None,
        # discretization runtime
        t_disc=None,
        # optimizer runtime
        t_solve=None,
        # total iteration duration
        t_iter=None,
        # propagation error too large
        ppgt_err=None) for _ in range(max_iter+1)]
    # initialize with initial guess
    iteration_data[0]['x'] = data['state_traj_init']
    iteration_data[0]['u'] = data['input_traj_init']

    # Execute successive convexification
    bar_x = data['state_traj_init']
    bar_u = data['input_traj_init']
    for j in range(1,max_iter+2):
        if j>max_iter:
            j -= 1
            print("\nreached max iterations")
            break
        t_iter = time.time()
        
        # Discretize
        t_disc = time.time()
        x_nl = discretize(data,bar_x,bar_u)
        t_disc = time.time()-t_disc
        
        # Solve
        bar_x,bar_u,L = solve_lrtop(data,bar_x,bar_u)
        t_solve = data['lrtop'].solver_stats.solve_time
        vc = np.row_stack([data['lrtop_var']['v'][k].value for k in range(N)])
        
        # Compute propagation error over [0,t_f]
        t_ppgt,x_ppgt,_ = csm.sim(state_init,bar_u,t_f,**ode_options['sim'])
        p_err = la.norm(x_ppgt[-1,:3]-bar_x[-1,:3],ord=np.inf)
        v_err = la.norm(x_ppgt[-1,3:6]-bar_x[-1,3:6],ord=np.inf)
        w_err = np.rad2deg(la.norm(x_ppgt[-1,10:]-bar_x[-1,10:],ord=np.inf))
        ang_err = np.rad2deg(2*np.arccos(tools.qmult(tools.qconj(
            bar_x[-1,6:10]),x_ppgt[-1,6:10])[0]))
        ppgt_err = (p_err>ppgt_tol['p'] or
                    v_err>ppgt_tol['v'] or
                    ang_err>ppgt_tol['ang'] or
                    w_err>ppgt_tol['w'])
        t_iter = time.time()-t_iter

        # Check if any state constraint is not satisfied in the propagated
        # dynamics
        x_ppgt_ct = siplt.interp1d(t_ppgt,x_ppgt,axis=0,assume_sorted=True,
                                   bounds_error=False,
                                   fill_value=(x_ppgt[0],x_ppgt[-1]))
        stc_imp_att = np.array([min(la.norm(x_ppgt_ct(t)[:3]-data['xf']['p'])-
                                    data['r_app'],0)*
                                (np.cos(0.5*np.deg2rad(data['ang_app']))-
                                 tools.qmult(tools.qconj(x_ppgt_ct(t)[6:10]),
                                             data['xf']['q'])[0])<=-eps_machine
                                for t in data['t_grid']])
        stc_app_cone = np.array([la.norm(x_ppgt_ct(t)[:3]-data['xf']['p'])*
                                 np.cos(np.deg2rad(data['gamma']))>=
                                 (x_ppgt_ct(t)[:3]-data['xf']['p']).T.dot(
                                     data['dock_axis'])+eps_machine
                                 for t in data['t_grid']])
        stc_app_cone[-1] = False # don't care about the final "singularity"
        x_viol = np.sum(stc_imp_att+stc_app_cone)
        
        # Save iteration data
        lb_viol = sum([np.sum(np.logical_and(
            bar_u[k]<csm.t_pulse_min-eps_machine,
            bar_u[k]>eps_machine)) for k in range(N)])
        iteration_data[j]['x'] = bar_x
        iteration_data[j]['x_nl'] = x_nl
        iteration_data[j]['t_ppgt'] = t_ppgt
        iteration_data[j]['x_ppgt'] = x_ppgt
        iteration_data[j]['u'] = bar_u
        iteration_data[j]['vc'] = vc
        iteration_data[j]['lb_viol'] = lb_viol
        iteration_data[j]['x_viol'] = x_viol
        iteration_data[j]['cost'] = L
        iteration_data[j]['J_tr'] = data['lrtop_cost']['J_tr'].value
        iteration_data[j]['J_vc'] = data['lrtop_cost']['J_vc'].value
        iteration_data[j]['t_disc'] = t_disc
        iteration_data[j]['t_solve'] = t_solve
        iteration_data[j]['t_iter'] = t_iter
        iteration_data[j]['ppgt_err'] = ppgt_err
        
        # Recover from input lower-bound constraint STC locking
        tbl_action = ''
        if ppgt_err:
            bar_x,bar_u,tbl_action = solution_reset(
                data,iteration_data,j,bar_x,bar_u)
            
        # Print progress        
        tbl_iteration = '%d%s'%(j,'!' if ppgt_err else '')
        tbl_trust_region = '%.2e'%(data['lrtop_cost']['J_tr'].value/w_tr)
        tbl_virtual_control = '%.2e'%(data['lrtop_cost']['J_vc'].value/w_vc)
        tbl_cost = '%.4f'%(L)
        tbl_lb_viol = '%d'%(lb_viol)
        tbl_x_viol = '%d'%(x_viol)
        tbl_p_err = '%.2e'%(p_err)
        tbl_v_err = '%.2e'%(v_err)
        tbl_ang_err = '%.2e'%(ang_err)
        tbl_w_err = '%.2e'%(w_err)
        tbl_row = [tbl_iteration,tbl_trust_region,tbl_virtual_control,tbl_cost,
                   tbl_lb_viol,tbl_x_viol,tbl_p_err,tbl_v_err,tbl_ang_err,
                   tbl_w_err,tbl_action]
        tools.print_table(tbl_row,tbl_col_size)
        
        # Check for convergence
        if j>1 and not ppgt_err and lb_viol==0:
            L_prev = iteration_data[j-1]['cost']
            rel_dJ = abs((L-L_prev)/L_prev)
            if data['use_stop_crit'] and rel_dJ<=dJ_tol:
                print("\nconverged")
                break

    # Save the solution
    t_sol = data['t_grid']
    p_sol = bar_x[:,:3].T
    v_sol = bar_x[:,3:6].T
    q_sol = bar_x[:,6:10].T
    omega_sol = bar_x[:,10:13].T
    u_sol = bar_u.T
    J_sol = L

    # Save the result of propagating the optimal input through the non-linear
    # dynamics
    p_ppgt = x_ppgt[:,:3].T
    v_ppgt = x_ppgt[:,3:6].T
    q_ppgt = x_ppgt[:,6:10].T
    w_ppgt = x_ppgt[:,10:13].T

    sol = dict(
        # Optimizer output
        optimizer=dict(t=t_sol,p=p_sol,v=v_sol,q=q_sol,w=omega_sol,u=u_sol,
                       cost=J_sol),
        # Non-linear propagated trajectory
        nl_prop=dict(t=t_ppgt,p=p_ppgt,v=v_ppgt,q=q_ppgt,w=w_ppgt,u=u_sol),
        # Iteration data
        history=iteration_data[:j+1])
    
    return sol

if __name__=='__main__':
    # Test scenario
    csm = apollo_csm.ApolloCSM()

    # CSM initial state
    p_csm_0 = np.array([0.,0.,0.])
    v_csm_0 = np.array([0.,0.,0.])
    q_csm_0 = tools.rpy2q(phi=0.,theta=180.,psi=0.)
    w_csm_0 = np.deg2rad(np.array([0.,0.,0.]))

    # LM state
    p_lm = np.array([20.,0.,0.])
    v_lm = np.array([0.,0.,0.])
    q_lm = tools.rpy2q(phi=0.,theta=180.,psi=0.)
    w_lm = np.array([0.,0.,0.])

    # Time of flight and control interval durations
    t_f = 150. # [s] Rendezvous duration
    t_silent = 2. # [s] control interval

    # Scaling for weights
    w_tr_ref = 1e3
    N_ref = 75.

    N = int(t_f/t_silent) # Temporal grid density
    
    inputs = dict(x0 = dict(p=p_csm_0,v=v_csm_0,q=q_csm_0,w=w_csm_0),
                  lm = dict(p=p_lm,v=v_lm,q=q_lm,w=w_lm),
                  N = N,
                  t_f = t_f,
                  t_pulse_max = csm.t_pulse_max,
                  max_iter = 50,
                  w_tr = 1e3,
                  w_vc = 1e7,
                  ppgt_tol = dict(p=1e-2,v=1e-3,ang=0.5,w=1e-2),
                  dJ_tol = 1e-3,
                  use_stop_crit = False,
                  v_dock = 0.1,
                  r_app = 4.,
                  ang_app = 2.,
                  gamma = 30.,
                  thrusters_off = [],
                  # thrusters_off = [i for i in range(csm.M) if
                  #                  'D' in csm.i2thruster[i]],
                  solver_options = dict(solver=cvx.MOSEK,verbose=False),
                  ode_options = dict(
                      dltv=dict(method='RK45'),
                      sim=dict(method='RK45',rtol=1e-6,atol=1e-10)))
    
    outputs = solve_rendezvous(csm,inputs)

    result = dict(solver_input=[inputs],solver_output=[outputs])
    with open('data/tf_%d_tc_%d_v2.pkl'%(t_f,t_silent),'wb') as f:
        pickle.dump(result,f)
