"""
Closed-loop vertical (z) position controller.

A PD law on z error. local_current_scale already re-balances Fz against
weight at the live pose each step, but it doesn't notice z drifting off
z_stable; this controller does. Fz is reachable through the coils,
unlike yaw. Returns Fx = Fy = 0 and zero torque so it sums cleanly in
CombinedController.
"""

from dataclasses import dataclass

import numpy as np

from control.base import Controller, Wrench


@dataclass
class ZTarget:
    # added by Wilf
    """Desired pod height (m)."""
    z: float


class ZPDController(Controller):
    # added by Wilf
    """
    Fz = -kp*(z - z_target) - kd*vz, saturated to |Fz| <= force_limit.
    No integral term (same reason as LateralPDController). kp in N/m,
    kd in N/(m/s).
    """

    def __init__(self, target, kp, kd, force_limit):
        self.target = target
        self.kp = kp
        self.kd = kd
        self.force_limit = force_limit

    def set_target(self, target):
        self.target = target

    def compute(self, t, state, dt):
        ez = state.z0 - self.target.z
        Fz = -self.kp * ez - self.kd * state.vz0

        if abs(Fz) > self.force_limit and self.force_limit > 0:
            Fz = self.force_limit if Fz > 0 else -self.force_limit

        return Wrench(force=np.array([0.0, 0.0, Fz]), torque=np.zeros(3))
