"""
Quaternion / SO(3) helpers for the attitude dynamics. No project
imports. Euler angles stay the setup/display form; the dynamics is
integrated in quaternions, which renormalise cheaply and don't hit
gimbal lock.

Convention: q = (w, x, y, z), Hamilton product, right-handed,
body-to-world, so quat_to_matrix(q) matches rotation_matrix_from_euler:
v_world = R @ v_body.
"""

import numpy as np


def quat_identity():
    # added by Wilf
    return np.array([1.0, 0.0, 0.0, 0.0])


def quat_normalize(q):
    # added by Wilf
    """q / |q|. The ODE solvers don't hold the unit norm, so call this
    after every step."""
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_to_matrix(q):
    # added by Wilf
    """Rotation matrix R with v_world = R @ v_body. q need not be
    normalised first."""
    w, x, y, z = quat_normalize(q)

    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R):
    # added by Wilf
    """
    Inverse of quat_to_matrix, by Shepperd's method (stable across the
    whole rotation range where the bare trace formula isn't). Returns a
    unit (w, x, y, z).
    """
    R = np.asarray(R, dtype=float)
    tr = R[0, 0] + R[1, 1] + R[2, 2]

    # four ways to recover q, each dividing by a different sqrt; take the
    # branch whose divisor is largest (the trace, or whichever diagonal
    # entry dominates) so the denominator stays well clear of zero
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    return quat_normalize(np.array([w, x, y, z]))


def quat_multiply(q1, q2):
    # added by Wilf
    """Hamilton product q1 * q2: applies q2 first, then q1."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conjugate(q):
    # added by Wilf
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_derivative(q, omega_body):
    # added by Wilf
    """Quaternion kinematics qdot = 0.5 * q * (0, omega_body), omega_body
    in the body frame."""
    omega_quat = np.array([0.0, omega_body[0], omega_body[1], omega_body[2]])
    return 0.5 * quat_multiply(q, omega_quat)


def skew(v):
    # added by Wilf
    """Hat map: (3,) vector to a (3,3) skew-symmetric matrix, with
    skew(v) @ w == cross(v, w)."""
    return np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def vee(M):
    # added by Wilf
    """Vee map: inverse of skew() for the (anti)symmetric part of M."""
    return np.array([M[2, 1], M[0, 2], M[1, 0]])
