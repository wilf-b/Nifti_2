"""
Closed-loop pod dynamics: a controller and allocator recompute the track
currents from the pod's own state as it moves.

There's no precomputed field grid here, unlike run_dynamics. solve_ivp
can't take a discrete control update mid-integration, so this is a manual
zero-order-hold loop: over each [t, t+control_dt) interval the currents
are set once (baseline schedule, then controller, then allocator) and
held while solve_ivp integrates across it. The field still varies
continuously with the pod's motion inside the interval.

New file; baseline_code.py had no feedback control.
"""

import numpy as np
from scipy.integrate import solve_ivp

from geometry.collection import Collection
from geometry.transforms import rotation_matrix_from_euler
from physics.field_query import field_from_collection, field_gradient_at_point
from track.control import assign_currents_for_pod
from physics.dipole import dipole_moment
from simulation.dynamics import DynamicsState, _rigid_body_derivatives
from control.base import Controller
from control.allocation import CurrentAllocator
from config import SimConfig


class _ClosedLoopResult:
    # added by Wilf
    """Minimal OdeResult-alike so plot_trajectory works unchanged."""
    def __init__(self, t, y):
        self.t = t
        self.y = y
        self.t_events = []


def run_closed_loop_dynamics(pod, pod_pts, track, controller, allocator,
                             initial_state, cfg, t_span, control_dt,
                             max_step=0.001, current_scale=1.0):
    # added by Wilf
    """
    Simulate pod rigid-body dynamics under live closed-loop current
    control.

    pod supplies mass, inertia tensor and dipole moment; pod_pts is from
    generate_pod_base_points(pod). track holds ElectroMagnet rows and has
    its currents mutated in place each step. controller returns the
    correction Wrench, allocator realises it as per-row dI. cfg and
    current_scale pass through to assign_currents_for_pod; current_scale
    defaults to 1.0 (the uncalibrated schedule, which doesn't levitate -
    see simulation/calibration.py). t_span is (t_start, t_end) s,
    control_dt the zero-order-hold period, max_step the solver step in an
    interval.

    Returns an object with .t, .y (12, n_steps) and .t_events. There is
    no field-bounds safety event, so a misbehaving controller can drive
    the pod into a field singularity near a coil.
    """
    m      = pod.TotalMass
    I_inv  = np.linalg.inv(pod.compute_inertia_tensor())
    m_body = dipole_moment(pod, pod_pts)

    t0, t_end = t_span
    state = np.array(initial_state.as_list(), dtype=float)

    t_chunks = []
    y_chunks = []

    t = t0
    while t < t_end - 1e-12:
        t_next = min(t + control_dt, t_end)

        x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz = state
        R       = rotation_matrix_from_euler(roll, pitch, yaw)
        m_world = R @ m_body

        # live baseline at the pod's position, then the correction on top
        pod_xy = np.hypot(x, y)
        assign_currents_for_pod(track, pod_xy=pod_xy, pod_total_mass=pod.TotalMass,
                                config=cfg, current_scale=current_scale)

        dyn_state = DynamicsState(
            x0=x, y0=y, z0=z, vx0=vx, vy0=vy, vz0=vz,
            roll0=roll, pitch0=pitch, yaw0=yaw, wx0=wx, wy0=wy, wz0=wz,
        )
        wrench = controller.compute(t, dyn_state, control_dt)
        allocator.allocate(wrench, track, state[:3], m_world)

        def dynamics(tt, ss):
            pt    = ss[:3]
            B     = field_from_collection(track, pt)[0]
            gradB = field_gradient_at_point(track, pt)
            return _rigid_body_derivatives(ss, B, gradB, m_body, m, I_inv)

        sol_seg = solve_ivp(dynamics, (t, t_next), state, max_step=max_step)

        t_chunks.append(sol_seg.t)
        y_chunks.append(sol_seg.y)

        state = sol_seg.y[:, -1]
        t     = sol_seg.t[-1]

    return _ClosedLoopResult(np.concatenate(t_chunks), np.concatenate(y_chunks, axis=1))
