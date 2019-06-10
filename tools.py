"""
Utility functions. Quaternion algebra based on [1].

[1] Joan Sola, "Quaternion kinematics for the error-state Kalman filter",
    https://arxiv.org/abs/1711.02508.

D. Malyuta -- ACL, University of Washington
B. Acikmese -- ACL, University of Washington

Copyright 2019 University of Washington. All rights reserved.
"""

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.integrate as sintgt

def qmult(p,q):
    """
    Quaternion multiplication p*q.
    Reference: (12) in arxiv:1711.02508

    Parameters
    ----------
    p : np.ndarray
        Left quaternion as [qw,qx,qy,qz].
    q : np.ndarray
        Right quaternion as [qw,qx,qy,qz].

    Returns
    -------
    r : np.ndarray
        Resultant quaternion.
    """
    pw,px,py,pz = p
    qw,qx,qy,qz = q
    r = np.array([pw*qw-px*qx-py*qy-pz*qz,
                  pw*qx+px*qw+py*qz-pz*qy,
                  pw*qy-px*qz+py*qw+pz*qx,
                  pw*qz+px*qy-py*qx+pz*qw])
    return r

def rqpmat(q):
    """
    Computes the right-quaternion-product matrix. Returns q_R such that
    qmult(p,q)=q_R*p.

    Parameters
    ----------
    q : np.ndarray
        Quaternion as [qw,qx,qy,qz].

    Returns
    -------
    q_R : np.ndarray
        Right-quaternion-product matrix.
    """
    qw,qx,qy,qz = q
    q_R = np.array([[qw,-qx,-qy,-qz],
                    [qx,qw,qz,-qy],
                    [qy,-qz,qw,qx],
                    [qz,qy,-qx,qw]])
    return q_R

def qconj(q):
    """
    Conjugate quaternion.

    Parameters
    ----------
    q :np.ndarray
        Original quaternion as [qw,qx,qy,qz].

    Returns
    -------
    qc : np.ndarray
        Conjugated quaternion.
    """
    qc = q.copy()
    qc[1:] *= -1.
    return qc

def qpure(r):
    """
    Convert r vector to a pure quaternion or a pure quaternion to a vector.

    Parameters
    ----------
    r : np.ndarray
        R^3 vector or quaternion as [qw,qx,qy,qz].

    Returns
    -------
    q : np.ndarray
        Pure quaternion [0,r] if vector, or vector part of quaternion.
    """
    if r.size==3:
        # Vector to pure quaternion
        q = np.array([0,r[0],r[1],r[2]])
    else:
        # Pure quaternion to vector
        q = r[1:]
    return q

def rotate(v,q):
    """
    Rotate vector v by quaternion q.

    Parameters
    ----------
    v : np.ndarray
        Vector in {A} frame.
    q : np.ndarray
        Quaternion transformation from {A} frame to {B} frame, as [qw,qx,qy,qz].

    Returns
    -------
    w : np.ndarray
        Vector in {B} frame.
    """
    w = qpure(qmult(q,qmult(qpure(v),qconj(q))))
    return w

def rot2q(R):
    """
    Convert rotation matrix to quaternion.

    Parameters
    ----------
    R : np.ndarray
        Rotation matrix.

    Returns
    -------
    q : np.ndarray
        Equivalent quaternion as [qw,qx,qy,qz].
    """
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr>0:
        S = np.sqrt(tr+1.)*2. # S=4*qw 
        qw = 0.25*S
        qx = (R[2,1]-R[1,2])/S
        qy = (R[0,2]-R[2,0])/S 
        qz = (R[1,0]-R[0,1])/S 
    elif (R[0,0] > R[1,1]) and (R[0,0] > R[2,2]):
        S = np.sqrt(1.0+R[0,0]-R[1,1]-R[2,2])*2 # S=4*qx 
        qw = (R[2,1]-R[1,2])/S
        qx = 0.25*S
        qy = (R[0,1]+R[1,0])/S 
        qz = (R[0,2]+R[2,0])/S 
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0+R[1,1]-R[0,0]-R[2,2])*2 # S=4*qy
        qw = (R[0,2]-R[2,0])/S
        qx = (R[0,1]+R[1,0])/S 
        qy = 0.25*S
        qz = (R[1,2]+R[2,1])/S 
    else:
        S = np.sqrt(1.0+R[2,2]-R[0,0]-R[1,1])*2 # S=4*qz
        qw = (R[1,0]-R[0,1])/S
        qx = (R[0,2]+R[2,0])/S
        qy = (R[1,2]+R[2,1])/S
        qz = 0.25*S
    q = np.array([qw,qx,qy,qz])
    cleanup_small_values(q)
    return q

def q2rpy(q):
    """
    Convert quaternion to roll, pitch, yaw in Tait-Bryan convention.

    Parameters
    ----------
    q : np.ndarray
        Equivalent quaternion as [qw,qx,qy,qz].

    Returns
    -------
    phi : float
        Roll in **degrees**.
    theta : float
        Pitch in **degrees**.
    psi : float
        Yaw in **degrees**.
    """
    phi = np.arctan2(2.*(q[0]*q[1]+q[2]*q[3]),1.-2.*(q[1]**2+q[2]**2))
    theta = np.arcsin(2*(q[0]*q[2]-q[3]*q[1]))
    psi = np.arctan2(2.*(q[0]*q[3]+q[1]*q[2]),1.-2.*(q[2]**2+q[3]**2))

    phi = np.rad2deg(phi)
    theta = np.rad2deg(theta)
    psi = np.rad2deg(psi)
    
    return phi,theta,psi

def rpy2q(phi,theta,psi):
    """
    Convert (roll,pitch,yaw) in Tait-Bryan convention to quaternion.

    Parameters
    ----------
    phi : float
        Roll in **degrees**.
    theta : float
        Pitch in **degrees**.
    psi : float
        Yaw in **degrees**.

    Returns
    -------
    q : np.ndarray
        Equivalent quaternion as [qw,qx,qy,qz].
    """
    cd = lambda u: np.cos(np.deg2rad(u))
    sd = lambda u: np.sin(np.deg2rad(u))
    Rx = lambda u: np.array([[1.,0.,0.],[0.,cd(u),-sd(u)],[0.,sd(u),cd(u)]])
    Ry = lambda u: np.array([[cd(u),0.,sd(u)],[0.,1.,0.],[-sd(u),0.,cd(u)]])
    Rz = lambda u: np.array([[cd(u),-sd(u),0.],[sd(u),cd(u),0.],[0.,0.,1.]])
    R = Rx(phi).dot(Ry(theta)).dot(Rz(psi)) # rotated frame --> original frame
    q = rot2q(R)
    return q

def skew(x):
    """
    Create skew symmetric matrix from x.

    Parameters
    ----------
    x : np.ndarray
        R^3 vector.

    Returns
    -------
    S : np.ndarray
        R^{3x3} skew-symmetric matrix.
    """
    S = np.array([[0.,-x[2],x[1]],
                  [x[2],0.,-x[0]],
                  [-x[1],x[0],0.]])
    return S

def slerp(q0,q1,tspan,tol=1e-6):
    """
    Spherical linear interpolation (SLERP) from quaternion q0 to q1.

    Parameters
    ----------
    q0 : np.ndarray
        Initial unit quaternion as [qw,qx,qy,qz].
    q1 : np.ndarray
        Target unit quaternion as [qw,qx,qy,qz].
    tspan : np.ndarray
        Times at which to interpolate the quaternion. Must be in [0,1].
    tol : float, optional
        Below this value, the quaternions are considered too close to each
        other and linear interpolation is applied.

    Returns
    -------
    qi : np.ndarray
        Matrix of intermediate quaternions whose row k is the interpolated
        quaternion at time tspan[k].
    omega : np.ndarray
        Angular velocity [rad/s] of the rotation (constant by construction of
        SLERP). None if tspan has only one element.
    """
    q0 = q0.copy()
    q1 = q1.copy()
    if q0.dot(q1)<0.:
        # Make sure SLERP takes the short path
        q1 *= -1.
    dq = qmult(qconj(q0),q1) # error quaternion
    dqv_norm = la.norm(dq[1:])
    angle = 2*np.arctan2(dqv_norm,dq[0])
    axis = (
        # error rotation axis
        dq[1:]/dqv_norm if dqv_norm>np.finfo(float).eps else
        # any axis will do if q0==q1
        np.array([1.,0.,0.]))
    if dqv_norm<tol:
        # q0 and q1 are too close, linearly interpolate and normalize instead
        qi = np.outer(1-tspan,q0)+np.outer(tspan,q1)
        qi /= np.array([la.norm(qi,axis=1)]).T
    else:
        # Apply SLERP
        qi = np.row_stack([qmult(q0,np.insert(axis*np.sin(t*angle/2.),
                                              0,np.cos(t*angle/2.)))
                           for t in tspan])
    trange = tspan[-1]-tspan[0] if tspan.size>1 else None
    omega = angle/trange*axis if trange is not None else None
    return qi,omega

def odeint(f,x0,t,**odeopts):
    """
    Integrate an ODE.

    Parameters
    ----------
    f : callable
        Dynamics oracle f(t,x) where t (type: float) is the current time and x
        (type: np.ndarray) is the state.
    x0 : np.ndarray
        The initial state.
    t : np.ndarray
        Times at which to store the solution.
    **odeopts : keyword arguments, optional
        Any option in the documentation of scipy.integrate.solve_ivp.
    
    Returns
    -------
    x : np.ndarray
        The state trajectory as a matrix whose row k is the integrated state at
        time t[k].
    """
    x = sintgt.solve_ivp(fun=f,
                         t_span=[t[0],t[-1]],
                         y0=x0,
                         t_eval=t,
                         **odeopts)['y'].T
    return x

def bounding_box(trajectory):
    """
    Compute the bounding box of a trajectory. The minimum sidelength of the box
    in any dimension is unity.

    Parameters
    ----------
    trajectory : np.ndarray
        The trajectory to bound within the box, as an array whose row k is the
        trajectory value at step k.

    Returns
    -------
    box : np.ndarray
        An array whose element k is the bounding box sidelength (symmetric)
        bounds the trajectory's column k, i.e.
        trajectory[:,k] \in [-box[k],box[k]].
    """
    box = np.max(np.abs(trajectory),axis=0)
    cleanup_small_values(box,val=1.,tol=1e-10)
    return box

def cleanup_small_values(x,val=0.,tol=None):
    """
    Remove small values from x. **Modifies x**.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    val : float, optional
        What value to set the small values to.
    tol : float, optional
        Tolerance to use (default is machine epsilon).

    Returns
    -------
    x : np.ndarray
        Same array, but with values smaller than machine epsilon cleaned up.
    """
    tol = np.finfo(float).eps if tol is None else tol
    x[np.abs(x)<tol] = val
    return x

def gram_schmidt(u):
    """
    Compute orthogonal basis starting from the set of vectors u, via
    Gram-Schmidt process.

    Parameters
    ----------
    u : np.ndarray
        The starting set of vectors, where each row is one of the vectors. Can
        also be a one-dimensional array, i.e. a starting single vector.

    Returns
    -------
    B : np.ndarray
        The basis matrix whose row k is the k-th basis vector.
    """
    u = np.array([u]) if u.ndim==1 else u
    N = sla.null_space(u) #np.eye(n)-np.outer(u,u)/u.dot(u) # nullspace of n
    B = np.row_stack([u,N.T])
    for i in range(B.shape[0]):
        B[i] = B[i]-sum([B[i].dot(B[j])*B[j] for j in range(0,i)])
        B[i] /= la.norm(B[i])
    return B

def print_table(entries,spacing):
    """
    Print a row of a table with well-defined spacing.

    Parameters
    ----------
    entries : list
        List of entries to print.
    spacing : list
        List of spacing between the entries.
    """
    str_format = ' '.join(['{: >%d}'%(space) for space in spacing])
    print(str_format.format(*entries))

def get_values(data,*fields):
    """
    Gets all the values stored in data saved by test scripts, for all runs.

    Parameters
    ----------
    data : dict
        The complete data structure.
    *fields : list
        List of fields to access, going down the hierarchy in data.

    Returns
    -------
    values : list
        Values of the variable pointed to by fields, for each element of the
        lists in data.
    """
    which_io = fields[0]
    io = data[which_io]
    if which_io=='solver_input':
        which_var = fields[1]
        values = [io[i][which_var] for i in range(len(io))]
    else:
        which_out = fields[1]
        which_var = fields[2]
        if which_out!='history':
            values = [io[i][which_out][which_var] for i in range(len(io))]
        else:
            values = [[io[i][which_out][k][which_var]
                       for k in range(len(io[i][which_out]))]
                      for i in range(len(io))]
    return values

def log_grid(ax,axis,interval):
    for exponent in range(interval[0],interval[1]):
        options = dict(color='lightgray',linewidth=1)
        for mantissa in range(1,10):
            if mantissa==1:
                options['linestyle'] = '-'
            else:
                options['linestyle'] = '--'
            if axis=='x':
                ax.axvline(mantissa*10**exponent,**options)
            else:
                ax.axhline(mantissa*10**exponent,**options)
