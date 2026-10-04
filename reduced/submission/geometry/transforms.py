"""
Rotation and coordinate-transform helpers. No project imports, so it's
safe to pull in from anywhere. rotation_matrix_from_euler is from
baseline_code.py; euler_from_matrix is its inverse made for the comparison.
"""

import numpy as np


def rotation_matrix_from_euler(roll, pitch, yaw):
    """
    3x3 rotation matrix from intrinsic ZYX Euler angles (roll about X,
    pitch about Y, yaw about Z, radians). R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rx = np.array([[1,  0,   0 ],
                   [0,  cr, -sr],
                   [0,  sr,  cr]])

    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])

    return Rz @ Ry @ Rx


def euler_from_matrix(R):
    # added by Wilf
    """
    Recover (roll, pitch, yaw) in radians from
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll). Display and plotting only; the
    SO(3) path integrates quaternions and never calls this, so its
    gimbal-lock singularity at pitch = +-pi/2 doesn't matter.
    """
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll  = np.arctan2(R[2, 1], R[2, 2])
    yaw   = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def rotate_point_into_frame(point, origin, R):
    """
    Express a world-frame point in the local frame given by `origin` and
    rotation `R` (local -> world, so R.T is the inverse). Used to map
    sensor positions into the pod frame when the pod is tilted.
    """
    relative = np.asarray(point, dtype=float) - np.asarray(origin, dtype=float)
    return R.T @ relative
