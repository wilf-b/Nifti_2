"""
Pod rigid-body dynamics. This is the Euler-angle path only, from
baseline_code.py. For SO(3) attitudes, use the full package.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass

from geometry.collection import Collection
from geometry.transforms import rotation_matrix_from_euler
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
        return [self.x0, self.y0, self.z0,
                self.vx0, self.vy0, self.vz0,
                self.roll0, self.pitch0, self.yaw0,
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
    """
    F = m_world . grad(B), plus gravity on z.
    """
    F = m_world @ gradB
    F[2] -= m * G
    return F


def _rigid_body_derivatives(state, B, gradB, m_body, m, I_inv):
    """
    12-state rigid-body ODE right-hand side: translation from the dipole
    force, rotation from the magnetic torque (tau = m_world x B).

    state is [x,y,z,vx,vy,vz,roll,pitch,yaw,wx,wy,wz]. B is the (3,)
    world-frame field at the pod, gradB the (3,3) with
    gradB[i,j] = dB_i/dx_j. m_body is the pod dipole moment in the body
    frame, m the pod mass (kg), I_inv the world-frame inverse inertia
    tensor. Returns the 12 state derivatives.
    """
    x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = state

    R       = rotation_matrix_from_euler(roll, pitch, yaw)
    m_world = R @ m_body

    F   = _dipole_force(m_world, gradB, m)
    tau = np.cross(m_world, B)

    ax, ay, az = F / m
    alpha      = I_inv @ tau

    return [vx, vy, vz, ax, ay, az,
            wx, wy, wz, alpha[0], alpha[1], alpha[2]]


def run_dynamics(pod, pod_pts, B_all, x_data, y_data, z_data,
                 initial_state, t_span=(0.0, 2.0), max_step=0.001):
    """
    Simulate pod rigid-body dynamics in a precomputed magnetic field.
    Open-loop: the currents are baked into B_all and never change during
    integration.

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
