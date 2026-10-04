"""
Closed-loop lateral (x,y) position controller: a PD law on lateral
position/velocity error giving a horizontal force command. Fz and torque
are always zero; keeping the lateral correction out of Fz is the
allocator's job.
"""

from dataclasses import dataclass

import numpy as np

from control.base import Controller, Wrench


@dataclass
class LateralTarget:
    # added by Wilf
    """Desired pod position in the horizontal (x,y) plane."""
    x: float
    y: float


class LateralPDController(Controller):
    # added by Wilf
    """
    Fx = -kp*(x - x_target) - kd*vx, Fy likewise, saturated to
    |F| <= force_limit. No integral term (it would need anti-windup under
    the allocator's current-limit saturation). kp in N/m, kd in N/(m/s).
    """

    def __init__(self, target, kp, kd, force_limit):
        self.target = target
        self.kp = kp
        self.kd = kd
        self.force_limit = force_limit

    def set_target(self, target):
        self.target = target

    def compute(self, t, state, dt):
        ex = state.x0 - self.target.x
        ey = state.y0 - self.target.y

        Fx = -self.kp * ex - self.kd * state.vx0
        Fy = -self.kp * ey - self.kd * state.vy0

        force = np.array([Fx, Fy, 0.0])
        norm = np.linalg.norm(force[:2])
        if norm > self.force_limit and norm > 0:
            force[:2] *= self.force_limit / norm

        return Wrench(force=force, torque=np.zeros(3))
