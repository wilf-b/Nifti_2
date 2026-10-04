"""
Composes several Controllers into one by summing their Wrenches. The
lateral (force only) and attitude (torque only) controllers zero the axes
they don't own, so the sum is clean. Each sub-controller saturates
itself; shared current headroom is CurrentAllocator's problem.
"""

import numpy as np

from control.base import Controller, Wrench


class CombinedController(Controller):
    # added by Wilf
    """
    Sums the Wrench from each controller in controllers, all called with
    the same (t, state, dt). last_wrenches keeps each one's most recent
    output for diagnostics.
    """

    def __init__(self, controllers):
        self.controllers = controllers
        self.last_wrenches = []

    def compute(self, t, state, dt):
        wrenches = []
        force = np.zeros(3)
        torque = np.zeros(3)
        for controller in self.controllers:
            w = controller.compute(t, state, dt)
            wrenches.append(w)
            force = force + w.force
            torque = torque + w.torque

        self.last_wrenches = wrenches
        return Wrench(force=force, torque=torque)
