"""
Pod rigid-body dynamics.

run_dynamics is open-loop: it interpolates a field grid built for a
fixed current schedule, so it holds only while nothing changes the track
currents. run_closed_loop_dynamics evaluates the field live instead.
Both share _rigid_body_derivatives.

The Euler-angle path (DynamicsState, run_dynamics,
_rigid_body_derivatives) is from baseline_code.py; the quaternion/SO(3)
path is new for attitude control.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass, field

from geometry.collection import Collection
from geometry.transforms import rotation_matrix_from_euler
from geometry.quaternion import quat_identity, quat_to_matrix, matrix_to_quat, quat_derivative
from physics.dipole import dipole_moment
from config import G


@dataclass
class DynamicsState:
    """
    Initial conditions for the dynamics solver. Positions in metres,
    velocities in m/s, angles in radians, angular rates in rad/s.
    """
    x0:    float = 0.0
    y0:    float = 0.0
    z0:    float = 0.40
    vx0:   float = 0.0
    vy0:   float = 0.0
    vz0:   float = 0.0
    roll0: float = 0.0
    pitch0:float = 0.0
    yaw0:  float = 0.0
    wx0:   float = 0.0
    wy0:   float = 0.0
    wz0:   float = 0.0

    def as_list(self):
        # added by Wilf
        return [self.x0, self.y0, self.z0,
                self.vx0, self.vy0, self.vz0,
                self.roll0, self.pitch0, self.yaw0,
                self.wx0, self.wy0, self.wz0]


@dataclass
class DynamicsStateSO3:
    # added by Wilf
    """
    Initial conditions for the SO(3) solver. Translational fields as
    DynamicsState; attitude is a quaternion q0 = (w,x,y,z) and
    wx0/wy0/wz0 is real body-frame angular velocity (DynamicsState
    integrates those as roll_dot/pitch_dot/yaw_dot, exact only for small
    angles).
    """
    x0:  float = 0.0
    y0:  float = 0.0
    z0:  float = 0.40
    vx0: float = 0.0
    vy0: float = 0.0
    vz0: float = 0.0
    q0:  np.ndarray = field(default_factory=quat_identity)
    wx0: float = 0.0
    wy0: float = 0.0
    wz0: float = 0.0

    @classmethod
    def from_euler(cls, x0=0.0, y0=0.0, z0=0.40,
                   vx0=0.0, vy0=0.0, vz0=0.0,
                   roll0=0.0, pitch0=0.0, yaw0=0.0,
                   wx0=0.0, wy0=0.0, wz0=0.0):
        # added by Wilf
        """
        Build a DynamicsStateSO3 from Euler-angle parameters, for scripts
        that set up an initial tilt the Euler-angle way. wx0/wy0/wz0 must
        already be body-frame angular velocity.
        """
        q0 = matrix_to_quat(rotation_matrix_from_euler(roll0, pitch0, yaw0))
        return cls(x0=x0, y0=y0, z0=z0, vx0=vx0, vy0=vy0, vz0=vz0,
                    q0=q0, wx0=wx0, wy0=wy0, wz0=wz0)

    def as_list(self):
        # added by Wilf
        return [self.x0, self.y0, self.z0,
                self.vx0, self.vy0, self.vz0,
                self.q0[0], self.q0[1], self.q0[2], self.q0[3],
                self.wx0, self.wy0, self.wz0]


def _safe_gradient(F, dx, dy, dz):
    """Gradient with zero-padding on single-point axes."""
    dF_dx = np.gradient(F, dx, axis=0) if F.shape[0] > 1 else np.zeros_like(F)
    dF_dy = np.gradient(F, dy, axis=1) if F.shape[1] > 1 else np.zeros_like(F)
    dF_dz = np.gradient(F, dz, axis=2) if F.shape[2] > 1 else np.zeros_like(F)
    return dF_dx, dF_dy, dF_dz


def _build_interpolators(B_all, x_data, y_data, z_data):
    """
    RegularGridInterpolator objects for B and all 9 gradient components,
    as a dict keyed 'Bx', 'By', 'Bz', 'dBx_dx', and so on.
    """
    Bx, By, Bz = B_all

    dx = (x_data[-1] - x_data[0]) / (len(x_data) - 1) if len(x_data) > 1 else 1.0
    dy = (y_data[-1] - y_data[0]) / (len(y_data) - 1) if len(y_data) > 1 else 1.0
    dz = (z_data[-1] - z_data[0]) / (len(z_data) - 1) if len(z_data) > 1 else 1.0

    grid = (x_data, y_data, z_data)
    kw   = dict(bounds_error=False, fill_value=0.0)

    interps = {
        "Bx": RegularGridInterpolator(grid, Bx, **kw),
        "By": RegularGridInterpolator(grid, By, **kw),
        "Bz": RegularGridInterpolator(grid, Bz, **kw),
    }

    for name, F in [("Bx", Bx), ("By", By), ("Bz", Bz)]:
        gx, gy, gz = _safe_gradient(F, dx, dy, dz)
        interps[f"d{name}_dx"] = RegularGridInterpolator(grid, gx, **kw)
        interps[f"d{name}_dy"] = RegularGridInterpolator(grid, gy, **kw)
        interps[f"d{name}_dz"] = RegularGridInterpolator(grid, gz, **kw)

    return interps


def _dipole_force(m_world, gradB, m):
    # added by Wilf
    """
    F = m_world . grad(B), plus gravity on z. Shared by the Euler-angle
    and SO(3) paths so there's only one force law between them.
    """
    F = m_world @ gradB
    F[2] -= m * G
    return F


def _rigid_body_derivatives(state, B, gradB, m_body, m, I_inv):
    """
    12-state rigid-body ODE right-hand side: translation from the dipole
    force, rotation from the magnetic torque (tau = m_world x B). Shared
    by run_dynamics (grid-interpolated field) and run_closed_loop_dynamics
    (live field) so they can't diverge. A closed-loop controller only
    enters indirectly, by changing track.Magnets[i].current before B and
    gradB are evaluated; there is no separate controller-force term.

    state is [x,y,z,vx,vy,vz,roll,pitch,yaw,wx,wy,wz]. B is the (3,)
    world-frame field at the pod, gradB the (3,3) with
    gradB[i,j] = dB_i/dx_j (field_query.field_gradient_at_point's
    convention). m_body is the pod dipole moment in the body frame, m the
    pod mass (kg), I_inv the world-frame inverse inertia tensor
    (Collection.compute_inertia_tensor's convention). Returns the 12
    state derivatives.
    """
    x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = state

    R       = rotation_matrix_from_euler(roll, pitch, yaw)
    m_world = R @ m_body

    F   = _dipole_force(m_world, gradB, m)
    tau = np.cross(m_world, B)

    return _integrate_rigid_body_euler(state, F, tau, m, I_inv)


def _integrate_rigid_body_euler(state, F, tau, m, I_inv):
    """
    Euler-angle-rate integration tail: from total force and torque about
    the COM, the 12 state derivatives.
    """
    _, _, _, vx, vy, vz, _, _, _, wx, wy, wz = state

    ax, ay, az = F / m
    alpha      = I_inv @ tau

    return [vx, vy, vz, ax, ay, az,
            wx, wy, wz, alpha[0], alpha[1], alpha[2]]


def _rigid_body_derivatives_so3(state, B, gradB, m_body, m,
                               I_body, I_body_inv, angular_damping=0.0):
    # added by Wilf
    """
    13-state rigid-body ODE RHS with quaternion kinematics and Euler's
    equation, valid under large rotations where _rigid_body_derivatives'
    Euler-angle-rate form isn't. Shares _dipole_force with it.

    state is [x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz], w the body-frame
    angular velocity (rad/s). B is the (3,) world field, gradB the (3,3)
    with gradB[i,j] = dB_i/dx_j. I_body is the constant body-frame
    inertia and I_body_inv its inverse. angular_damping applies
    -angular_damping * omega_body as a stabilisation aid (default 0.0;
    the torque-free conservation test relies on that default). Returns
    the 13 state derivatives.
    """
    q       = np.array([state[6], state[7], state[8], state[9]])
    R       = quat_to_matrix(q)
    m_world = R @ m_body

    F         = _dipole_force(m_world, gradB, m)
    tau_world = np.cross(m_world, B)

    return _integrate_rigid_body_so3(
        state, F, tau_world, R, m, I_body, I_body_inv,
        angular_damping,
    )


def _integrate_rigid_body_so3(state, F, tau_world, R, m,
                             I_body, I_body_inv, angular_damping=0.0):
    # added by Wilf
    """
    Quaternion / Euler's-equation integration tail: total force and
    world-frame torque about the COM to the 13 state derivatives.
    """
    _, _, _, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = state
    q          = np.array([qw, qx, qy, qz])
    omega_body = np.array([wx, wy, wz])

    ax, ay, az = F / m

    tau_body = R.T @ tau_world

    omega_dot = I_body_inv @ (
        tau_body - np.cross(omega_body, I_body @ omega_body) - angular_damping * omega_body
    )
    qdot      = quat_derivative(q, omega_body)

    return [vx, vy, vz, ax, ay, az,
            qdot[0], qdot[1], qdot[2], qdot[3],
            omega_dot[0], omega_dot[1], omega_dot[2]]


def run_dynamics(pod, pod_pts, B_all, x_data, y_data, z_data,
                 initial_state, t_span=(0.0, 2.0), max_step=0.001):
    """
    Simulate pod rigid-body dynamics in a precomputed magnetic field.
    Open-loop: the currents are baked into B_all and never change during
    integration. run_closed_loop_dynamics is the feedback counterpart.

    pod supplies mass and inertia tensor. pod_pts is from
    generate_pod_base_points(pod) and sets the dipole moment's
    dV = TotalVolume / N. B_all is [Bx, By, Bz], each (Nx, Ny, Nz), from
    load_magnetic_field, and x_data/y_data/z_data are the grid coordinate
    arrays. t_span is (t_start, t_end) in seconds, max_step the solver
    step (s). Returns a scipy OdeResult with sol.t and sol.y[0..11].
    """
    interps = _build_interpolators(B_all, x_data, y_data, z_data)

    m      = pod.TotalMass
    I      = pod.compute_inertia_tensor()
    I_inv  = np.linalg.inv(I)
    m_body = dipole_moment(pod, pod_pts)

    x_min, x_max = x_data[0], x_data[-1]
    y_min, y_max = y_data[0], y_data[-1]
    z_min, z_max = z_data[0], z_data[-1]

    def out_of_bounds(t, s):
        return min(s[0]-x_min, x_max-s[0],
                   s[1]-y_min, y_max-s[1],
                   s[2]-z_min, z_max-s[2])

    out_of_bounds.terminal  = True
    out_of_bounds.direction = -1

    def dynamics(t, state):
        x, y, z = state[0], state[1], state[2]
        pt = (x, y, z)

        B = np.array([interps["Bx"](pt), interps["By"](pt), interps["Bz"](pt)]).flatten()
        gradB = np.array([
            [float(interps["dBx_dx"](pt)), float(interps["dBx_dy"](pt)), float(interps["dBx_dz"](pt))],
            [float(interps["dBy_dx"](pt)), float(interps["dBy_dy"](pt)), float(interps["dBy_dz"](pt))],
            [float(interps["dBz_dx"](pt)), float(interps["dBz_dy"](pt)), float(interps["dBz_dz"](pt))],
        ])

        return _rigid_body_derivatives(state, B, gradB, m_body, m, I_inv)

    return solve_ivp(
        dynamics,
        t_span,
        initial_state.as_list(),
        max_step   = max_step,
        events     = out_of_bounds,
    )
