"""
Position-space linearisation of the pod/track force plant, for sanity
checking the PD gains in control/. Not used in the dynamics integration,
which runs the exact nonlinear model.  No baseline_code.py equivalent.
"""

import numpy as np

from geometry.collection import Collection
from physics.field_query import field_gradient_at_point, field_from_collection
from track.control import assign_currents_for_pod
from config import SimConfig


def build_total_position_stiffness(
    track,
    pod_position,
    m_vec,
    pod_total_mass,
    config,
    current_scale=1.0,
    h=1e-4,
):
    # added by Wilf
    """
    Total-derivative force-vs-position Jacobian: re-solves the baseline
    schedule at each perturbed position, so it captures the baseline
    current's own dependence on pod_xy that
    build_position_stiffness_jacobian misses. Mutates and restores
    track.Magnets[i].current, leaving it set for pod_position's pod_xy on
    exit. current_scale is a float or a callable (pos, m_vec) -> float,
    same convention as simulation/closed_loop_so3.py. Returns a (3, 3)
    array in N/m.
    """
    pod_position = np.asarray(pod_position, dtype=float)
    J = np.zeros((3, 3))

    def _force_at(pos):
        pod_xy = float(np.hypot(pos[0], pos[1]))
        scale = current_scale(pos, m_vec) if callable(current_scale) else current_scale
        assign_currents_for_pod(track, pod_xy=pod_xy, pod_total_mass=pod_total_mass,
                                 config=config, current_scale=scale)
        grad = field_gradient_at_point(track, pos)
        return m_vec @ grad

    for k in range(3):
        step = np.zeros(3)
        step[k] = h
        F_plus  = _force_at(pod_position + step)
        F_minus = _force_at(pod_position - step)
        J[:, k] = (F_plus - F_minus) / (2 * h)

    # leave currents set for the expansion point, not the last perturbation
    _force_at(pod_position)

    return J
