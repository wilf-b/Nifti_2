"""
Support code for notebook 04. Drives the two attitude integrators in
simulation/dynamics.py with no field or dipole, i.e. a torque-free
rigid-body tumble, and gives back the conservation diagnostics.

The Euler-angle path integrates the body rates as roll/pitch/yaw rates and
drops the -omega x (I omega) term; the SO(3) path integrates a quaternion
and carries Euler's equation in full. The inertia tensor is a real pod
preset's, so the precession is physical.
"""

import numpy as np
from scipy.integrate import solve_ivp

from data_io.presets import load_pod_preset
from geometry.transforms import euler_from_matrix
from geometry.quaternion import quat_identity
from simulation.dynamics import _rigid_body_derivatives, _rigid_body_derivatives_so3


def pod_body_inertia(pod_name):
    """(pod, I_body, I_body_inv, mass) for a named preset, level attitude."""
    pod = load_pod_preset(pod_name)
    I_body = pod.compute_body_inertia_tensor()
    return pod, I_body, np.linalg.inv(I_body), pod.TotalMass


def run_tumble(I_body, I_body_inv, mass, omega0, t_span, mode,
               rtol=1e-10, atol=1e-13, max_step=np.inf):
    """
    Torque-free tumble from level attitude. `mode` is "euler" or "so3";
    omega0 is the initial body-frame angular velocity (rad/s). Returns the
    solve_ivp result. Rotational state rows are roll/pitch/yaw + wx,wy,wz
    for "euler", qw,qx,qy,qz + wx,wy,wz for "so3".
    """
    zb, zg, zm = np.zeros(3), np.zeros((3, 3)), np.zeros(3)
    w0 = [float(w) for w in omega0]

    if mode == "euler":
        state0 = [0.0] * 9 + w0
        rhs = lambda t, s: _rigid_body_derivatives(s, zb, zg, zm, mass, I_body_inv)
    elif mode == "so3":
        state0 = [0.0] * 6 + list(quat_identity()) + w0
        rhs = lambda t, s: _rigid_body_derivatives_so3(s, zb, zg, zm, mass, I_body, I_body_inv)
    else:
        raise ValueError(f"mode must be 'euler' or 'so3', got {mode!r}")

    return solve_ivp(rhs, t_span, state0, rtol=rtol, atol=atol,
                     max_step=max_step, dense_output=True)


def angular_momentum(rotations, omega_body, I_body):
    """(3, N) world-frame L(t) = R(t) . (I_body . omega_body(t))."""
    Iw = I_body @ omega_body
    return np.einsum("nij,jn->in", np.asarray(rotations), Iw)


def momentum_drift(L_world, L0):
    """(N,) relative drift |L_world(t) - L0| / |L0|."""
    return np.linalg.norm(L_world - L0[:, None], axis=0) / np.linalg.norm(L0)


def euler_angles(rotations):
    """(3, N) ZYX (roll, pitch, yaw) recovered from each R(t)."""
    return np.array([euler_from_matrix(R) for R in rotations]).T
