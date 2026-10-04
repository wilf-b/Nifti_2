"""
Closed-loop roll/pitch attitude controller: geometric PD on SO(3), roll
and pitch only. Yaw is left open-loop because torque about the pod's
moment axis is unreachable through the coils under pod_model="point".
The law is Proposition 1 of Lee, Leok and McClamroch, "Geometric
Tracking Control of a Quadrotor UAV on SE(3)", 49th IEEE CDC 2010
(arXiv:1003.2005). Force is always zero; gains k_R, k_omega are not
rig-calibrated.
"""

from dataclasses import dataclass

import numpy as np

from control.base import Controller, Wrench
from geometry.transforms import rotation_matrix_from_euler
from geometry.quaternion import quat_to_matrix, vee


@dataclass
class AttitudeTarget:
    # added by Wilf
    """Desired pod roll/pitch (rad). Yaw has no effect here."""
    roll: float
    pitch: float


class GeometricAttitudePDController(Controller):
    # added by Wilf
    """
    Geometric PD on SO(3), roll/pitch only:

        e_R      = 1/2 * vee(R_d^T R - R^T R_d)
        tau_body = -k_R * e_R - k_omega * omega_body

    The body-z (yaw-like) parts of e_R and omega are zeroed first so no
    torque is asked for about an unactuated axis; using body-z rather
    than world-z is a small-angle approximation that holds in the
    roll/pitch regime here. tau is rotated to world and saturated to
    |tau| <= torque_limit.
    """

    def __init__(self, target, k_R, k_omega, torque_limit):
        self.target = target
        self.k_R = k_R
        self.k_omega = k_omega
        self.torque_limit = torque_limit

        # last compute() call's intermediates, for inspecting
        # commanded-vs-saturated torque; overwritten every call
        self.last_e_R = np.zeros(3)
        self.last_omega_body = np.zeros(3)
        self.last_tau_body_precommand = np.zeros(3)
        self.last_tau_world = np.zeros(3)

    def set_target(self, target):
        self.target = target

    def compute(self, t, state, dt):
        R = quat_to_matrix(state.q0)
        R_d = rotation_matrix_from_euler(self.target.roll, self.target.pitch, 0.0)
        omega_body = np.array([state.wx0, state.wy0, state.wz0])

        e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
        e_omega = omega_body.copy()

        e_R[2] = 0.0   # body-z (yaw-like): structurally unactuated, don't chase it
        e_omega[2] = 0.0

        tau_body = -self.k_R * e_R - self.k_omega * e_omega
        tau = R @ tau_body   # body -> world, to match Wrench's convention

        self.last_e_R = e_R.copy()
        self.last_omega_body = omega_body.copy()
        self.last_tau_body_precommand = tau_body.copy()

        norm = np.linalg.norm(tau)
        if norm > self.torque_limit and norm > 0:
            tau *= self.torque_limit / norm

        self.last_tau_world = tau.copy()

        return Wrench(force=np.zeros(3), torque=tau)
