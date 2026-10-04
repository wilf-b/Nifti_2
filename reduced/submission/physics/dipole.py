"""
Two dipole models for the pod.

dipole_moment collapses the pod to one point dipole at the COM (what
baseline_code.py used). multi_dipole_force_torque keeps each magnet as
its own dipole and picks up a lever-arm term a single point can't
produce. That term is why yaw torque, which looks unreachable for a bare
m x B, is actually reachable through the coils. Lives in physics/ so
control/allocation.py can import it without a control -> simulation cycle.
"""

import numpy as np
from geometry.collection import Collection


def dipole_moment(pod, pod_pts):
    # added by Wilf
    """Single-point dipole moment for the whole pod: angle * dV * M, with
    dV = TotalVolume / len(pod_pts). Only pod_pts.shape[0] is used."""
    dV = pod.TotalVolume / pod_pts.shape[0]
    M  = pod.Magnets[0].magnetic_strength
    return pod.Magnets[0].angle * dV * M


def per_magnet_body_moments(pod):
    # added by Wilf
    """Each magnet's body-frame dipole moment, angle * volume *
    magnetic_strength, in pod.Magnets order. Uses each magnet's real
    volume, so the magnitudes won't match dipole_moment."""
    moments = []
    for magnet in pod.Magnets:
        moments.append(magnet.angle * magnet.volume * magnet.magnetic_strength)
    return moments


def total_dipole_moment_magnitude(pod):
    # added by Wilf
    """Sum of |m_i| over per_magnet_body_moments. A magnitude sanity check
    against dipole_moment before trusting multi-dipole numbers; no
    dynamics path uses it."""
    total = 0.0
    for moment in per_magnet_body_moments(pod):
        total += np.linalg.norm(moment)
    return float(total)


def multi_dipole_force_torque(pod, R, COM_world, B_fn, gradB_fn):
    # added by Wilf
    """
    Force and torque (about COM_world) on a rigid pod carrying several
    point dipoles, each at its own world position rather than collapsed
    to the COM. Per magnet:

        r_world = COM_world + R @ (magnet.position - pod.Position)
        m_world = R @ (magnet.angle * magnet.volume * magnet.magnetic_strength)
        F_i     = m_world . gradB_fn(r_world)
        tau_i   = cross(m_world, B_fn(r_world)) + cross(r_world - COM_world, F_i)

    The second torque term (the lever arm) is what a single COM dipole
    can't produce. R is world-from-body, COM_world the current COM; B_fn
    maps a point to B (3,), gradB_fn to gradB (3, 3) with
    gradB[i, j] = dB_i/dx_j. Gravity is not included here; the caller adds
    -m*G on z once on the total mass.
    """
    F_total   = np.zeros(3)
    tau_total = np.zeros(3)

    for magnet in pod.Magnets:
        r_body  = magnet.position - pod.Position
        r_world = COM_world + R @ r_body

        m_body_i  = magnet.angle * magnet.volume * magnet.magnetic_strength
        m_world_i = R @ m_body_i

        B_i     = B_fn(r_world)
        gradB_i = gradB_fn(r_world)

        F_i = m_world_i @ gradB_i
        F_total += F_i

        tau_total += np.cross(m_world_i, B_i)
        tau_total += np.cross(r_world - COM_world, F_i)

    return F_total, tau_total
