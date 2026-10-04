"""
Dipole moment calculation: the pod's magnetic dipole in the body frame.
From baseline_code.py, inlined in various loops.
"""

import numpy as np


def dipole_moment(pod, pod_pts):
    """
    Single-point dipole moment for the whole pod: angle * dV * M, with
    dV = TotalVolume / len(pod_pts). Ported from baseline_code.py's
    inline calculation.
    """
    dV = pod.TotalVolume / pod_pts.shape[0]
    M = pod.Magnets[0].magnetic_strength
    m_vec = pod.Magnets[0].angle * dV * M
    return m_vec
