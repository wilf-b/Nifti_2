"""
Shared fixtures and path bootstrap for the plain-assert test files. Not
a test file itself, and not in tests/run_all.py's list. Each test file
does `from _helpers import ...` after putting this directory on sys.path.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geometry.transforms import rotation_matrix_from_euler
from geometry.sampling import generate_pod_base_points
from geometry.quaternion import quat_normalize, quat_to_matrix
from track.geometry import TrackRow, track_rows_to_collection
from physics.field_query import field_at_points, field_gradient_at_point
from physics.dipole import dipole_moment
from data_io.presets import load_track_preset, load_pod_preset

# shared fixtures
INERTIA_DIAG = np.diag([0.002, 0.005, 0.008])          # asymmetric top
TILT_R = rotation_matrix_from_euler(0.2, 0.2, 0.0)      # a representative tilted pose
TILTED_POSE = np.array([0.005, 0.005, 0.040])
POD_POS = np.array([0.0, 0.0, 0.03])
M_VEC_TEST = np.array([0.0, 0.0, 1e-3])                 # simple upward test dipole


def synthetic_electro_track(current=0.05):
    """3 electro rows x 3 coils, spread across y, symmetric about y=0 (9 coils)."""
    track = track_rows_to_collection([
        TrackRow(0, -0.02, 3, 0.02, 0.005, 0.01, "electro", turns=200),
        TrackRow(1, 0.0, 3, 0.02, 0.005, 0.01, "electro", turns=200),
        TrackRow(2, 0.02, 3, 0.02, 0.005, 0.01, "electro", turns=200),
    ])
    for m in track.Magnets:
        m.current = current
    return track


def realized_force(track, pos, m_vec):
    """Sum field_gradient_at_point over the track's magnets, contracted with m_vec."""
    grad = np.zeros((3, 3))
    for mag in track.Magnets:
        grad += field_gradient_at_point(mag, pos)
    return m_vec @ grad


def realized_torque(track, pos, m_vec):
    """m x (sum of per-magnet B) at pos."""
    B = np.zeros(3)
    for mag in track.Magnets:
        B += field_at_points(mag, pos)[0]
    return np.cross(m_vec, B)


def load_classic_two_stacks():
    """(track, pod, pod_pts, m_body) for Classic Track / Two Stacks. Raises
    FileNotFoundError if the presets are missing, and the caller decides
    whether to skip."""
    track, _ = load_track_preset("Classic Track")
    pod = load_pod_preset("Two Stacks")
    pod_pts = generate_pod_base_points(pod)
    return track, pod, pod_pts, dipole_moment(pod, pod_pts)


def angmom_and_ke(state, I_body):
    """(world-frame L, KE) for an SO(3) state vector [.., q(6:10), omega(10:13)]."""
    q = quat_normalize(state[6:10])
    w = state[10:13]
    L_body = I_body @ w
    return quat_to_matrix(q) @ L_body, 0.5 * w @ L_body
