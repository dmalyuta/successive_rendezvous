"""
Apollo command service module (CSM) dynamical plant.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np
import numpy.linalg as la
import scipy.interpolate as siplt

import tools

class ApolloCSM:
    def __init__(self):
        self.set_parameters()

    def set_parameters(self):
        # Unit converter functions
        def convert_units(parameters,converter):
            """
            Converts the units of parameters using the converter oracle.

            Parameters
            ----------
            parameters : {float,np.array,dict}
                parameters.
            converter : callable
                Callable function converter(parameter) where parameter (type:
                float,np.array) is the parameter in its original units, which
                outputs the parameter in the units produced by the converter.

            Returns
            -------
            new_parameters : {float,np.array,dict}
                parameters in the new units.
            """
            if type(parameters) is dict:
                keys = parameters.keys()
                new_parameters = {k:None for k in keys}
                for key in keys:
                    new_parameters[key] = converter(parameters[key])
            else:
                new_parameters = converter(parameters)
            return new_parameters
        
        in2m = in2m = lambda x: x*0.0254 # inches to meters
        ft2slug2m2kg = lambda x: x*1.35581795 # slug*ft^2 to kg*m^2
        lb2kg = lambda x: x*0.453592 # lb to kg
        lbf2N = lambda x: x*4.448222 # lbf to N

        # Rotation primitives (angles to be given in **degrees**)
        cd = lambda angle: np.cos(np.deg2rad(angle))
        sd = lambda angle: np.sin(np.deg2rad(angle))
        R_x = lambda a: np.array([[1.,0.,0.],
                                  [0.,cd(a),-sd(a)],
                                  [0.,sd(a),cd(a)]])
        R_z = lambda a: np.array([[cd(a),-sd(a),0.],
                                  [sd(a),cd(a),0.],
                                  [0.,0.,1.]])

        # Total mass
        self.m = convert_units(66850.6,lb2kg)
        self.minv = 1./self.m

        # Inertia tensor in dynamical coordinates (i.e. body frame - same as
        # Apollo coordinates, but origin at center of mass)
        J_xx,J_yy,J_zz = 36324.,80036.,81701.
        J_xy,J_xz,J_yz = -2111.,273.,2268.
        self.J = np.array([[J_xx,-J_xy,-J_xz],
                           [-J_xy,J_yy,-J_yz],
                           [-J_xz,-J_yz,J_zz]])
        self.J = convert_units(self.J,ft2slug2m2kg)
        self.Jinv = la.inv(self.J)

        # Maximum thrust of each thruster
        self.thrust_max = convert_units(100.,lbf2N)

        # Minimum and maximum thrust pulse duration [s]
        self.t_pulse_min = 100e-3
        self.t_pulse_max = 500e-3

        ## CSM geometry below
        # {A} : CSM coordinate frame (centered behind CG)
        # {C} : CSM dynamical coordinate frame (centered at CG)
        
        # Thruster chamber root positions in quad frame
        Q_r_Q = dict(p_f=np.array([6.75,0.,0.]),
                     p_a=np.array([-6.75,0.,0.]),
                     r_f=np.array([0.94,0.,3.125]),
                     r_a=np.array([-0.94,0.,-3.125]))
        Q_r_Q = convert_units(Q_r_Q,in2m)

        # Thruster vectors in quad frame
        e_x = np.array([1.,0.,0.])
        e_z = np.array([0.,0.,1.])
        self.thruster_cant_angle = 10.
        Q_ = dict(p_f=R_z(self.thruster_cant_angle).dot(e_x),
                  p_a=R_z(-self.thruster_cant_angle).dot(-e_x),
                  r_f=R_x(self.thruster_cant_angle).dot(e_z),
                  r_a=R_x(-self.thruster_cant_angle).dot(-e_z))
        
        # Quad positions in RCS frame
        R_r_RQ = dict(A=np.array([958.97,0.,-83.56]),
                      B=np.array([958.97,83.56,0.]),
                      C=np.array([958.97,0.,83.56]),
                      D=np.array([958.97,-83.56,0.]))
        R_r_RQ = convert_units(R_r_RQ,in2m)

        # Center of mass position of CSM in Apollo frame
        A_r_AM = convert_units(np.array([933.9,5.0,4.7]),in2m)

        # RCS frame orientation wrt Apollo frame
        R_AR = R_x(-(7.+15./60.))

        # Quad frame orientations wrt RCS frame
        R_RQ = dict(A=R_x(-90.),B=R_x(0.),C=R_x(90.),D=R_x(180.))

        # Position of thrust chamber root and thrust vector of each thruster wrt
        # CG, in dynamical coordinates
        self.M = 4*4 # 4 thrusters per quad, 4 quads on SM
        self.thruster_position = np.empty((self.M,3)) # thruster positions
        self.thruster_vector = np.empty((self.M,3)) # thruster vectors
        self.i2thruster = {i:None for i in range(self.M)}
        self.thruster2i = dict()
        i = 0
        for quad in ['A','B','C','D']:
            for thruster in ['p_f','p_a','r_f','r_a']:
                # r = A_r_M(thruster # of quad #)
                self.thruster_position[i] = (R_AR.dot(
                    R_r_RQ[quad]+R_RQ[quad].dot(Q_r_Q[thruster]))-A_r_AM)
                self.thruster_vector[i] = -R_AR.dot(R_RQ[quad]).dot(Q_[thruster])
                self.i2thruster[i] = (quad,thruster)
                self.thruster2i[quad+' '+thruster] = i
                i += 1
        # precompute J^{-1}(r_i\cross F_i) terms that will be useful for the
        # dynamics
        self.Jinv_rxF = [self.Jinv.dot(np.cross(
            self.thruster_position[i],self.thruster_vector[i]*self.thrust_max))
                         for i in range(self.M)]

        # LM interface position in Apollo frame
        A_r_AI = convert_units(np.array([1110.25,0.,0.]),in2m)

        # LM interface position of CSM, wrt CG, in CSM dynamical coordinates
        C_r_CI = A_r_AI-A_r_AM
        self.r_dock_csm = C_r_CI

        ## LM geometry below
        # {E} : LM coordinate frame (centered behind CG)
        # {L} : LM dynamical coordinate frame (centered at CG)
        
        # CSM interface in LM frame
        E_r_EI = convert_units(np.array([312.5,0.,0.]),in2m)

        # Docking orientation (Apollo frame to LM frame)
        angle_dock = np.deg2rad(60.)
        R_EA = np.array([[-1.,0.,0.],
                         [0.,np.cos(angle_dock),np.sin(angle_dock)],
                         [0.,np.sin(angle_dock),-np.cos(angle_dock)]])
        q_EA = tools.rot2q(R_EA)
        self.q_dock = q_EA

        # Center of mass position of LM in LM frame
        A_r_AL = convert_units(np.array([1238.2,-0.6,0.8]),in2m)
        E_r_EL = tools.rotate(A_r_AL-A_r_AM-C_r_CI,q_EA)+E_r_EI

        # CSM interface position of LM, wrt CG, in LM dynamical coordinates
        L_r_LI = E_r_EI-E_r_EL
        self.r_dock_lm = L_r_LI

        ## Other quantities
        # DLTV creation conglomerated state for dynamic quantities
        self.dltv_z0 = []
        self.dltv_map = dict()
        def add_term(initial_value,name):
            """
            Add new term to concatenated state.

            Parameters
            ----------
            initial_value : np.ndarray
                Initial value of the dynamic variable.
            name : str
                Name to use of the dynamica variable.
            """
            idx_start = len(self.dltv_z0)
            self.dltv_z0 += initial_value.flatten().tolist()
            idx_end = len(self.dltv_z0)
            self.dltv_map[name] = dict(idx=range(idx_start,idx_end),
                                       sz=initial_value.shape)

        def get_term(z,name):
            """Extract term from conglomerated state."""
            return z[self.dltv_map[name]['idx']].reshape(
                self.dltv_map[name]['sz'])

        add_term(np.eye(3),'Phi_w')
        add_term(np.zeros(3),'Psi_r')
        add_term(np.zeros((3,3)),'Psi_u')
        add_term(np.eye(4),'Phi_q')
        add_term(np.zeros((4,3)),'Psi_E')
        add_term(np.zeros(4),'Psi_q')
        add_term(np.zeros(4),'Psi_Er')
        add_term(np.zeros((4,3)),'Psi_Ew')
        add_term(np.zeros((3,4)),'Psi_vq')
        add_term(np.zeros((3,3)),'Psi_vqw')
        add_term(np.zeros((3,3)),'Psi_vqu')
        add_term(np.zeros(3),'Psi_vrp')
        add_term(np.zeros(3),'Psi_vrppp')
        add_term(np.zeros(3),'Psi_vrpppp')
        add_term(np.zeros(3),'Psi_pvr')
        add_term(np.zeros((3,16)),'Psi_F')
        add_term(np.zeros((3,16)),'Psi_prF')
        add_term(np.zeros((3,4)),'Psi_pvq')
        add_term(np.zeros(3),'Psi_pvrppp')
        add_term(np.zeros((3,3)),'Psi_pw2')
        add_term(np.zeros(3),'Psi_pEr2')
        add_term(np.zeros((3,3)),'Psi_pvqu')
        
        self.dltv_z0 = np.array(self.dltv_z0)
        self.dltv_zget = get_term

    def __thrust_signal(self,u,t):
        """
        On/off thrust signal.

        Parameters
        ----------
        u : float
            Impulse duration.
        t : float
            The current time (starting from zero).

        Returns
        -------
        on : bool
            ``True`` if the thruster is firing.
        """
        eps_mach = np.finfo(float).eps # machine epsilon
        on = t<=u+eps_mach and u>=eps_mach
        return on

    def __compute_t_grid(self,u,t_f,n_sub):
        """
        Build the time grid, accounting for thruster impulse falling edges.

        Parameters
        ----------
        u : np.ndarray
            Array of control inputs where row k gives the k-th input. The inputs
            are assumed to be spaced at equal time intervals on [0,t_f] such
            that the last input is applied one time interval **before** t_f.
        t_f : float
            Simulation duration.
        n_sub : int, optiona
            Node density over the control interval (including start and end
            points). To accomodate thruster impulses, time nodes are added for
            the falling edge of each impulse. Must be >1. Default is to output
            just the control interval start and end states (i.e. n_sub==2).

        Returns
        -------
        t : np.ndarray
            List of simulation times for each control interval.
        mask : list
            List whose k-th element is the index of the element in t and row in
            x corresponding to a falling edge of a thrust impulse.
        """
        # 
        N = u.shape[0]
        t_silent = t_f/N # Control interval duration
        t,mask = [],[]
        for k in range(N):
            # Goal: create a time grid which has nodes at impulse falling edges
            # mesh the vanilla time grid with impulse pulse widths
            _t = np.concatenate([np.linspace(0.,t_silent,n_sub),u[k]])
            # sort the times
            idx = np.argsort(_t)
            _t = _t[idx]
            # find unique time nodes
            t_k,unique_idx = np.unique(_t,return_inverse=True)
            # mask to go from input i -> which time node holds its falling edge?
            mask_k = unique_idx[np.argsort(idx)][n_sub:]
            t.append(t_k)
            mask.append(mask_k)
        return t,mask

    def linearize(self,q,w,u,t):
        """
        Compute linearization about reference.

        Parameters
        ----------
        q : np.ndarray
            Reference quaternion as [qw,qx,qy,qz].
        w : np.ndarray
            Reference angular velocity.
        u : np.ndarray
            Reference input.
        t : float
            Reference time.
        
        Returns
        -------
        A : dict
            Block terms of the linearized dynamics' A matrix.
        r : dict
            Block terms of the linearized dynamics' residual term.
        """
        qw,qv = q[0],q[1:]
        input_on = [self.__thrust_signal(u[i],t) for i in range(self.M)]
        thrust = [self.thruster_vector[i]*self.thrust_max*input_on[i]
                  for i in range(self.M)]
        I3 = np.eye(3)
        A = dict(ww=-self.Jinv.dot(tools.skew(w).dot(self.J)-
                                   tools.skew(self.J.dot(w))),
                 qq=0.5*np.block([[0.,-w.T],[np.array([w]).T,-tools.skew(w)]]),
                 qw=0.5*np.row_stack([-qv.T,qw*I3+tools.skew(qv)]),
                 vq=2.*self.minv*sum([np.column_stack([
                     qw*thrust[i]+np.cross(qv,thrust[i]),
                     qv.dot(thrust[i])*I3+np.outer(qv,thrust[i])-
                     np.outer(thrust[i],qv)-qw*tools.skew(thrust[i])])
                                      for i in range(self.M)]),
                 pv=I3)
        r = dict(w=-self.Jinv.dot(np.cross(w,self.J.dot(w)))-A['ww'].dot(w),
                 q=0.5*tools.qmult(q,tools.qpure(w))-A['qq'].dot(q)-
                 A['qw'].dot(w))
        return A,r

    def sim(self,x0,u,t_f,n_sub=2,normalize_q=True,**odeopts):
        """
        Propagate the input through the non-linear dynamics.

        Parameters
        ----------
        x0 : np.ndarray
            Initial state.
        u : np.ndarray
            Array of control inputs where row k gives the k-th input. The inputs
            are assumed to be spaced at equal time intervals on [0,t_f] such
            that the last input is applied one time interval **before** t_f.
        t_f : float
            Simulation duration.
        n_sub : int, optiona
            Node density over the control interval (including start and end
            points). To accomodate thruster impulses, time nodes are added for
            the falling edge of each impulse. Must be >1. Default is to output
            just the control interval start and end states (i.e. n_sub==2).
        normalize_q : bool, optional
            If ``True``, normalize the unit quaternion at each time step.
        **odeopts : keyword arguments
            Arguments to pass to tools.odeint.

        Returns
        -------
        t : np.ndarray
            The array of simulation times.
        x : np.ndarray
            The integrated state trajectory where row k is the state at time
            t_sim[k].
        mask : list
            List whose k-th element is the index of the element in t and row in
            x corresponding to a falling edge of a thrust impulse.
        """
        N = u.shape[0]
        t,mask = self.__compute_t_grid(u,t_f,n_sub)
        x = [np.empty((t[k].size,13)) for k in range(N)]
        
        # Simulate
        def dxdt(t,x,u):
            """
            Compute state time derivative using the unforced non-linear
            dynamics.

            Parameters
            ----------
            t : float
                The current time in the control interval (starting from zero).
            x : np.ndarray
                The current state.
            u : np.ndarray
                The array of pulse widths for each thruster.

            Returns
            -------
            : np.ndarray
                State time derivative.
            """
            input_on = [self.__thrust_signal(u[i],t) for i in range(self.M)]
            v,q,w = x[3:6],x[6:10],x[10:13] # velocity,quaternion,angular rate
            if normalize_q:
                q /= la.norm(q) # re-normalize
            dpdt = v
            dvdt = self.minv*sum([tools.rotate(
                self.thruster_vector[i]*self.thrust_max*input_on[i],q)
                                  for i in range(self.M)])
            dqdt = 0.5*tools.qmult(q,tools.qpure(w))
            dwdt = (sum([self.Jinv_rxF[i]*input_on[i] for i in range(self.M)])-
                    self.Jinv.dot(np.cross(w,self.J.dot(w))))
            return np.concatenate([dpdt,dvdt,dqdt,dwdt])

        for k in range(N): # control interval k
            grid_sz = t[k].size
            x[k][0] = x0 if k==0 else x[k-1][-1]
            for i in range(grid_sz-1):
                # By restarting the integration between each node, and knowing
                # that there is a node at each impulse falling edge, we
                # explicitly avoid the "stiffness" of the dynamics induced by
                # the jumps in thrust values. Hence, a typical solver like RKF45
                # can be used
                t_eval = np.array([t[k][i],t[k][i+1]])
                x_traj = tools.odeint(lambda t,x: dxdt(t,x,u[k]),x[k][i],t_eval,
                                      **odeopts)
                x[k][i+1] = x_traj[-1]
        
        _t,_m = [],[]
        for k in range(N):
            t_last = 0. if k==0 else _t[-1]
            _m += (len(_t)+mask[k]).tolist()
            _t += (t_last+t[k]).tolist()
        t = np.array(_t)
        mask = np.array(_m)
        x = np.row_stack(x)
        
        return t,x,mask

    def compute_fuel_used(self,u,t_f):
        """
        Compute fuel consumed by the input trajectory.

        Parameters
        ----------
        u : np.ndarray
            Array of control inputs where row k gives the k-th input. The inputs
            are assumed to be spaced at equal time intervals on [0,t_f] such
            that the last input is applied one time interval **before** t_f.
        t_f : float
            Simulation duration.
        **odeopts : keyword arguments
            Arguments to pass to tools.odeint.

        Returns
        -------
        t : np.ndarray
            Time grid of the integration.
        fuel : np.ndarray
            Fuel consumption history.
        """
        N = u.shape[0]
        t,mask = self.__compute_t_grid(u,t_f,n_sub=10)
        x = [np.empty((t[k].size,1)) for k in range(N)]
        
        # Simulate
        def dfdt(t,u):
            """
            Compute state time derivative using the unforced non-linear
            dynamics.

            Parameters
            ----------
            t : float
                The current time in the control interval (starting from zero).
            u : np.ndarray
                The array of pulse widths for each thruster.

            Returns
            -------
            : np.ndarray
                Fuel consumption rate.
            """
            input_on = [self.__thrust_signal(u[i],t) for i in range(self.M)]
            N = np.sum(input_on)
            # Table 4.3-1 of SNA-8-D-027(I), Rev 3
            rate_1 = 0.16828277 # [kg/s] fuel consumption rate for one thruster
            return rate_1*N**2

        odeopts = dict(method='RK45')
        for k in range(N): # control interval k
            grid_sz = t[k].size
            x[k][0] = np.array([0.]) if k==0 else x[k-1][-1]
            for i in range(grid_sz-1):
                # By restarting the integration between each node, and knowing
                # that there is a node at each impulse falling edge, we
                # explicitly avoid the "stiffness" of the dynamics induced by
                # the jumps in thrust values. Hence, a typical solver like RKF45
                # can be used
                t_eval = np.array([t[k][i],t[k][i+1]])
                x_traj = tools.odeint(lambda t,x: dfdt(t,u[k]),x[k][i],t_eval,
                                      **odeopts)
                x[k][i+1] = x_traj[-1]
        
        _t = []
        for k in range(N):
            t_last = 0. if k==0 else _t[-1]
            _t += (t_last+t[k]).tolist()
        t = np.array(_t)
        fuel = np.row_stack(x)
        
        return t,fuel

    def dltv(self,t_star,x_star,u_star,**odeopts):
        """
        Compute discrete-time LTV system using impulse thrusts and the
        trajectory {t_star,x_star,u_star} as reference.

        Parameters
        ----------
        t_star : np.ndarray
            The temporal nodes of the reference trajectory.
        x_star : np.ndarray
            The states of the reference trajectory, as a matrix whose row k is
            the state at time t_star[k].
        u_star : np.ndarray
            The inputs of the reference trajectory, as a matrix whose row k is
            the input at time t_star[k].
        **odeopts : keyword arguments
            Arguments to pass to tools.odeint.

        Returns
        -------
        Ad : dict
            Discrete update A[k] matrices.
        Bd : dict
            Discrete update B[k] matrices.
        rd : dict
            Discrete update r[k] residual terms.
        x_nl : list
            List of trajectories obtained from propagating the non-linear
            dynamics using u_star, with re-setting for each control (silent
            time) segment. Element k of the list is the k-th control segment.
        """        
        def dzdt(z,t_ref,x_ref,u_ref):
            """
            The dynamics of the "concatenated" state z which allows to compute
            the discrete-time LTV updated matrices A, B and residual r.
            
            Parameters
            ----------
            z : np.ndarray
                The current concantenated state.
            t_ref : float
                Reference time about which to linearize the non-linear dynamics.
            x_ref : np.ndarray
                Reference state about which to linearize the non-linear
                dynamics.
            u_ref : np.ndarray
                Reference input about which to linearize the non-linear
                dynamics.
            
            Returns
            -------
            : np.ndarray
                The concatenated state's time derivative.
            """
            # Extract needed terms
            Phi_w = self.dltv_zget(z,'Phi_w')
            Phi_w_inv = la.inv(Phi_w)
            Phi_q = self.dltv_zget(z,'Phi_q')
            Phi_q_inv = la.inv(Phi_q)
            Psi_r = self.dltv_zget(z,'Psi_r')
            Psi_u = self.dltv_zget(z,'Psi_u')
            Psi_q = self.dltv_zget(z,'Psi_q')
            Psi_E = self.dltv_zget(z,'Psi_E')
            Psi_Ew = self.dltv_zget(z,'Psi_Ew')
            Psi_Er = self.dltv_zget(z,'Psi_Er')
            Psi_vrp = self.dltv_zget(z,'Psi_vrp')
            Psi_vrppp = self.dltv_zget(z,'Psi_vrppp')
            Psi_vrpppp = self.dltv_zget(z,'Psi_vrpppp')
            Psi_F = self.dltv_zget(z,'Psi_F')
            Psi_vq = self.dltv_zget(z,'Psi_vq')
            Psi_vqw = self.dltv_zget(z,'Psi_vqw')
            Psi_vqu = self.dltv_zget(z,'Psi_vqu')
            q_ref = x_ref[6:10]
            w_ref = x_ref[10:13]
            # Linearize about current propagated trajectory value
            A,r = self.linearize(q_ref,w_ref,u_ref,t_ref)
            # Compute time derivatives
            Phi_w_dot = A['ww'].dot(Phi_w)
            Psi_r_dot = Phi_w_inv.dot(r['w'])
            Psi_u_dot = Phi_w_inv
            Phi_q_dot = A['qq'].dot(Phi_q)
            E = Phi_q_inv.dot(A['qw']).dot(Phi_w)
            Psi_E_dot = E
            Psi_q_dot = Phi_q_inv.dot(r['q'])
            Psi_Er_dot = E.dot(Psi_r)
            Psi_Ew_dot = E.dot(Psi_u)
            Psi_vq_dot = A['vq'].dot(Phi_q)
            Psi_vqw_dot = Psi_vq_dot.dot(Psi_E)
            Psi_vqu_dot = Psi_vq_dot.dot(Psi_Ew)
            Psi_vrp_dot = -A['vq'].dot(q_ref)
            Psi_vrppp_dot = Psi_vq_dot.dot(Psi_q)
            Psi_vrpppp_dot = Psi_vq_dot.dot(Psi_Er)
            Psi_pvr_dot = Psi_vrp
            Psi_F_dot = np.column_stack([
                tools.rotate(self.thruster_vector[i]*self.thrust_max,q_ref)
                for i in range(self.M)])
            Psi_prF_dot = Psi_F
            Psi_pvq_dot = Psi_vq
            Psi_pvrppp_dot = Psi_vrppp
            Psi_pw2_dot = Psi_vqw
            Psi_pEr2_dot = Psi_vrpppp
            Psi_pvqu_dot = Psi_vqu
            # Create the conflomerate state time derivative
            dzdt = np.empty(self.dltv_z0.shape)
            dzdt[self.dltv_map['Phi_w']['idx']] = Phi_w_dot.flatten()
            dzdt[self.dltv_map['Psi_r']['idx']] = Psi_r_dot.flatten()
            dzdt[self.dltv_map['Psi_u']['idx']] = Psi_u_dot.flatten()
            dzdt[self.dltv_map['Phi_q']['idx']] = Phi_q_dot.flatten()
            dzdt[self.dltv_map['Psi_E']['idx']] = Psi_E_dot.flatten()
            dzdt[self.dltv_map['Psi_q']['idx']] = Psi_q_dot.flatten()
            dzdt[self.dltv_map['Psi_Er']['idx']] = Psi_Er_dot.flatten()
            dzdt[self.dltv_map['Psi_Ew']['idx']] = Psi_Ew_dot.flatten()
            dzdt[self.dltv_map['Psi_vq']['idx']] = Psi_vq_dot.flatten()
            dzdt[self.dltv_map['Psi_vqw']['idx']] = Psi_vqw_dot.flatten()
            dzdt[self.dltv_map['Psi_vqu']['idx']] = Psi_vqu_dot.flatten()
            dzdt[self.dltv_map['Psi_vrp']['idx']] = Psi_vrp_dot.flatten()
            dzdt[self.dltv_map['Psi_vrppp']['idx']] = Psi_vrppp_dot.flatten()
            dzdt[self.dltv_map['Psi_vrpppp']['idx']] = Psi_vrpppp_dot.flatten()
            dzdt[self.dltv_map['Psi_pvr']['idx']] = Psi_pvr_dot.flatten()
            dzdt[self.dltv_map['Psi_F']['idx']] = Psi_F_dot.flatten()
            dzdt[self.dltv_map['Psi_prF']['idx']] = Psi_prF_dot.flatten()
            dzdt[self.dltv_map['Psi_pvq']['idx']] = Psi_pvq_dot.flatten()
            dzdt[self.dltv_map['Psi_pvrppp']['idx']] = Psi_pvrppp_dot.flatten()
            dzdt[self.dltv_map['Psi_pw2']['idx']] = Psi_pw2_dot.flatten()
            dzdt[self.dltv_map['Psi_pEr2']['idx']] = Psi_pEr2_dot.flatten()
            dzdt[self.dltv_map['Psi_pvqu']['idx']] = Psi_pvqu_dot.flatten()
            return dzdt

        # Prepare storage variables
        N = len(t_star)-1
        Ad = {k:None for k in range(N)}
        Bd = {k:None for k in range(N)}
        rd = {k:None for k in range(N)}

        # Propagate the input through the non-linear dynamics to generate a
        # reference trajectory
        # **trick**: re-set integration to x_star[k] at every interval
        t_nl,x_nl,x_nl_ct,mask_nl = [],[],[],[]
        for k in range(N):
            _t,_x,_mask = self.sim(x0=x_star[k],
                                   u=np.array([u_star[k]]),
                                   t_f=t_star[k+1]-t_star[k],
                                   normalize_q=False,
                                   **odeopts)
            t_nl.append(_t)
            x_nl.append(_x)
            x_nl_ct.append(siplt.interp1d(_t,_x,axis=0,assume_sorted=True,
                                          bounds_error=False,
                                          fill_value=(_x[0],_x[-1])))
            mask_nl.append(_mask)

        # Compute discrete-time update for each interval
        for k in range(N):
            # Integrate the dynamic components
            pulse = u_star[k]
            z = tools.odeint(lambda t,z: dzdt(z,t,x_nl_ct[k](t),pulse),
                             x0=self.dltv_z0.copy(),t=t_nl[k],**odeopts)
            # Extract the dynamic components' values at t_star[k+1]
            j = [mask_nl[k][i] for i in range(self.M)]
            Phi_w_T = self.dltv_zget(z[-1],'Phi_w')
            Phi_w = [self.dltv_zget(z[j[i]],'Phi_w') for i in range(self.M)]
            Phi_w_inv = [la.inv(Phi_w[i]) for i in range(self.M)]
            Psi_r = self.dltv_zget(z[-1],'Psi_r')
            Psi_u = [self.dltv_zget(z[j[i]],'Psi_u') for i in range(self.M)]
            Phi_q_T = self.dltv_zget(z[-1],'Phi_q')
            Phi_q = [self.dltv_zget(z[j[i]],'Phi_q') for i in range(self.M)]
            Psi_E_T = self.dltv_zget(z[-1],'Psi_E')
            Psi_E = [self.dltv_zget(z[j[i]],'Psi_E') for i in range(self.M)]
            Psi_Ew = [self.dltv_zget(z[j[i]],'Psi_Ew') for i in range(self.M)]
            Psi_q_T = self.dltv_zget(z[-1],'Psi_q')
            Psi_Er_T = self.dltv_zget(z[-1],'Psi_Er')
            Psi_vq_T = self.dltv_zget(z[-1],'Psi_vq')
            Psi_vq = [self.dltv_zget(z[j[i]],'Psi_vq') for i in range(self.M)]
            Psi_vqw = [self.dltv_zget(z[j[i]],'Psi_vqw') for i in range(self.M)]
            Psi_vqw_T = self.dltv_zget(z[-1],'Psi_vqw')
            Psi_vqu = [self.dltv_zget(z[j[i]],'Psi_vqu') for i in range(self.M)]
            Psi_vrp_T = self.dltv_zget(z[-1],'Psi_vrp')
            Psi_vrppp_T = self.dltv_zget(z[-1],'Psi_vrppp')
            Psi_vrpppp_T = self.dltv_zget(z[-1],'Psi_vrpppp')
            Psi_pvr_T = self.dltv_zget(z[-1],'Psi_pvr')
            Psi_F = [self.dltv_zget(z[j[i]],'Psi_F')[:,i]
                     for i in range(self.M)]
            Psi_prF = [self.dltv_zget(z[j[i]],'Psi_prF')[:,i]
                       for i in range(self.M)]
            Psi_pvq_T = self.dltv_zget(z[-1],'Psi_pvq')
            Psi_pvq = [self.dltv_zget(z[j[i]],'Psi_pvq') for i in range(self.M)]
            Psi_pvrppp_T = self.dltv_zget(z[-1],'Psi_pvrppp')
            Psi_pw2_T = self.dltv_zget(z[-1],'Psi_pw2')
            Psi_pw2 = [self.dltv_zget(z[j[i]],'Psi_pw2') for i in range(self.M)]
            Psi_pvqu = [self.dltv_zget(z[j[i]],'Psi_pvqu')
                        for i in range(self.M)]
            Psi_pEr2_T = self.dltv_zget(z[-1],'Psi_pEr2')
            # Compute the discrete-time update components
            thrust_vec_inertial = [tools.rotate(
                self.thruster_vector[i]*self.thrust_max,
                x_nl[k][mask_nl[k][i]][6:10]) for i in range(self.M)]
            A_ww = Phi_w_T
            B_w = A_ww.dot(np.column_stack([
                Phi_w_inv[i].dot(self.Jinv_rxF[i]) for i in range(self.M)]))
            r_w = (
                # r_w_p
                A_ww.dot(Psi_r)+
                #r_w_pp
                A_ww.dot(sum([(Psi_u[i]-Phi_w_inv[i]*pulse[i]).dot(
                    self.Jinv_rxF[i]) for i in range(self.M)])))
            A_qq = Phi_q_T
            A_qw = A_qq.dot(Psi_E_T)
            B_q = A_qq.dot(np.column_stack([
                (Psi_E_T-Psi_E[i]).dot(Phi_w_inv[i]).dot(self.Jinv_rxF[i])
                for i in range(self.M)]))
            r_q = (
                # r_q_p
                A_qq.dot(Psi_q_T)+
                # r_q_pp
                A_qq.dot(Psi_Er_T)+
                # r_q_ppp
                A_qq.dot(sum([(Psi_Ew[i]+(Psi_E_T-Psi_E[i]).dot(
                    Psi_u[i]-Phi_w_inv[i]*pulse[i])).dot(self.Jinv_rxF[i])
                              for i in range(self.M)])))
            A_vq = Psi_vq_T
            A_vw = Psi_vqw_T
            B_v = (
                # B_v_p
                self.minv*np.column_stack([
                    thrust_vec_inertial[i] for i in range(self.M)])+
                # B_v_pp
                np.column_stack([
                    (Psi_vqw_T-Psi_vqw[i]-
                     (Psi_vq_T-Psi_vq[i]).dot(Psi_E[i])).dot(Phi_w_inv[i]).dot(
                         self.Jinv_rxF[i])
                    for i in range(self.M)]))
            r_v = (
                # r_v_p
                Psi_vrp_T+
                # r_v_pp
                self.minv*(sum(Psi_F)-sum([thrust_vec_inertial[i]*pulse[i]
                                           for i in range(self.M)]))+
                # r_v_ppp
                Psi_vrppp_T+
                # r_v_pppp
                Psi_vrpppp_T+
                # r_v_ppppp
                sum([(Psi_vqu[i]+(Psi_vq_T-Psi_vq[i]).dot(
                    Psi_Ew[i]-Psi_E[i].dot(Psi_u[i]))+
                      (Psi_vqw_T-Psi_vqw[i]).dot(Psi_u[i])-
                      (Psi_vqw_T-Psi_vqw[i]-
                       (Psi_vq_T-Psi_vq[i]).dot(Psi_E[i])).dot(Phi_w_inv[i])*
                      pulse[i]).dot(self.Jinv_rxF[i])
                    for i in range(self.M)]))
            T = t_star[k+1]-t_star[k]
            A_pv = np.eye(3)*T
            A_pq = Psi_pvq_T
            A_pw = Psi_pw2_T
            B_p = (
                # B_p_p
                self.minv*np.column_stack([(T-pulse[i])*thrust_vec_inertial[i]
                                           for i in range(self.M)])+
                # B_p_pp
                np.column_stack([
                    (Psi_pw2_T-Psi_pw2[i]-(T-pulse[i])*Psi_vqw[i]-
                     (Psi_pvq_T-Psi_pvq[i]-(T-pulse[i])*Psi_vq[i]).dot(
                         Psi_E[i])).dot(Phi_w_inv[i]).dot(self.Jinv_rxF[i])
                    for i in range(self.M)]))
            r_p = (
                # r_p_p
                Psi_pvr_T+
                # r_p_pp
                self.minv*sum([Psi_prF[i]+(T-pulse[i])*(
                    Psi_F[i]-thrust_vec_inertial[i]*pulse[i])
                               for i in range(self.M)])+
                # r_p_ppp
                Psi_pvrppp_T+
                # r_p_pppp
                Psi_pEr2_T+
                # r_p_ppppp
                sum([(Psi_pvqu[i]+(T-pulse[i])*Psi_vqu[i]+
                      (Psi_pvq_T-Psi_pvq[i]-
                       (T-pulse[i])*Psi_vq[i]).dot(Psi_Ew[i])+
                      (Psi_pw2_T-Psi_pw2[i]-(T-pulse[i])*Psi_vqw[i]-
                       (Psi_pvq_T-Psi_pvq[i]-(T-pulse[i])*Psi_vq[i]).dot(
                           Psi_E[i])).dot(Psi_u[i]-Phi_w_inv[i]*pulse[i])).dot(
                               self.Jinv_rxF[i])for i in range(self.M)]))
            # Concatenate
            Ad[k] = np.block([[np.eye(3),A_pv,A_pq,A_pw],
                              [np.zeros((3,3)),np.eye(3),A_vq,A_vw],
                              [np.zeros((4,6)),A_qq,A_qw],
                              [np.zeros((3,10)),A_ww]])
            Bd[k] = np.row_stack([B_p,B_v,B_q,B_w])
            rd[k] = np.concatenate([r_p,r_v,r_q,r_w])

        return Ad,Bd,rd,x_nl

