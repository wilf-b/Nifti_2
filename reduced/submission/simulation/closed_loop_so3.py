"""
Closed-loop pod dynamics with SO(3) (roll/pitch) attitude control, the
SO(3) counterpart to simulation/closed_loop.py. Same zero-order-hold
loop; the differences are a 13-element state (quaternion + body-frame
angular velocity), _rigid_body_derivatives_so3, and a quaternion
renormalise at every control-interval boundary.
"""

import numpy as np
from scipy.integrate import solve_ivp

from geometry.collection import Collection
from geometry.quaternion import quat_to_matrix, quat_normalize
from physics.field_query import field_from_collection, field_gradient_at_point
from track.control import assign_currents_for_pod
from physics.dipole import dipole_moment
from simulation.dynamics import DynamicsStateSO3, _rigid_body_derivatives_so3
from control.base import Controller, Wrench
from control.allocation import CurrentAllocator
from config import SimConfig


class _ClosedLoopResultSO3:
    # added by Wilf
    """Minimal OdeResult-alike so plot_trajectory_so3 works unchanged.
    .y rows are [x,y,z, vx,vy,vz, qw,qx,qy,qz, wx,wy,wz]."""
    def __init__(self, t, y):
        self.t = t
        self.y = y
        self.t_events = []


def run_closed_loop_dynamics_so3(pod, pod_pts, track, controller, allocator,
                                 initial_state, cfg, t_span, control_dt,
                                 max_step=0.001, current_scale=1.0,
                                 on_step=None, angular_damping=0.0):
    # added by Wilf
    """
    Same as run_closed_loop_dynamics bar the SO(3) state and derivatives.
    allocator needs include_torque=True for a torque command to reach the
    currents (not enforced here).

    pod supplies mass, body-frame inertia and dipole moment; pod_pts from
    generate_pod_base_points(pod). t_span is (t_start, t_end) s,
    control_dt the zero-order-hold period, max_step the solver step.

    current_scale is a float or a callable (pod_position, m_world) ->
    float recomputed each step so the baseline can track (x, y) drift;
    default 1.0 is the uncalibrated schedule. No free-fall check, so keep
    initial_state.z0 near the track without proper scaling.

    on_step(t, state, wrench, track), if given, runs once per step after
    the controller/allocator, for diagnostics only. angular_damping goes
    straight to _rigid_body_derivatives_so3 (a stabilisation aid, not an
    eddy-current model).

    Returns an object with .t, .y (13, n_steps) and .t_events.
    """
    m          = pod.TotalMass
    I_body     = pod.compute_body_inertia_tensor()
    I_body_inv = np.linalg.inv(I_body)
    m_body     = dipole_moment(pod, pod_pts)

    t0, t_end = t_span
    state = np.array(initial_state.as_list(), dtype=float)
    state[6:10] = quat_normalize(state[6:10])

    t_chunks = []
    y_chunks = []

    t = t0
    while t < t_end - 1e-12:
        t_next = min(t + control_dt, t_end)

        x, y, z, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = state
        q       = np.array([qw, qx, qy, qz])
        R       = quat_to_matrix(q)
        m_world = R @ m_body

        # live baseline at the pod's position, then the correction on top
        pod_xy = np.hypot(x, y)
        scale_now = current_scale(state[:3], m_world) if callable(current_scale) else current_scale
        assign_currents_for_pod(track, pod_xy=pod_xy, pod_total_mass=pod.TotalMass,
                                config=cfg, current_scale=scale_now)

        dyn_state = DynamicsStateSO3(
            x0=x, y0=y, z0=z, vx0=vx, vy0=vy, vz0=vz,
            q0=q, wx0=wx, wy0=wy, wz0=wz,
        )
        wrench = controller.compute(t, dyn_state, control_dt)
        allocator.allocate(wrench, track, state[:3], m_world)

        if on_step is not None:
            on_step(t, state.copy(), wrench, track)

        def dynamics(tt, ss):
            pt    = ss[:3]
            B     = field_from_collection(track, pt)[0]
            gradB = field_gradient_at_point(track, pt)
            return _rigid_body_derivatives_so3(
                ss, B, gradB, m_body, m, I_body, I_body_inv, angular_damping=angular_damping,
            )

        sol_seg = solve_ivp(dynamics, (t, t_next), state, max_step=max_step)

        t_chunks.append(sol_seg.t)
        y_chunks.append(sol_seg.y)

        state = sol_seg.y[:, -1]
        state[6:10] = quat_normalize(state[6:10])
        t = sol_seg.t[-1]

    return _ClosedLoopResultSO3(np.concatenate(t_chunks), np.concatenate(y_chunks, axis=1))
