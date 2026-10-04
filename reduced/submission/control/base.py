"""
The Wrench command type and the Controller interface. A controller works
out the force/torque the pod needs and returns it as a world-frame
Wrench; turning that into coil currents is the allocator's job.
Setpoints live on the concrete controller, since a lateral target and an
attitude target have nothing in common.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Wrench:
    # added by Wilf
    """A world-frame force/torque command: force [Fx,Fy,Fz] (N),
    torque [Tx,Ty,Tz] (N*m)."""
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.force = np.asarray(self.force, dtype=float)
        self.torque = np.asarray(self.torque, dtype=float)


class Controller:
    # added by Wilf
    """Base class for anything that turns pod state into a commanded Wrench."""

    def compute(self, t, state, dt):
        """Return the commanded Wrench for simulation time t (s), the current
        pod state, and dt seconds since the last control update."""
        raise NotImplementedError
