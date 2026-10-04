"""
Magnet geometry classes: pure data containers for the shape and material
of a magnet. There's no field_at() method here, computing a field from
geometry is physics and lives in physics/field_query.py.

Magnet is the base (geometry plus a containment test); PermMagnet adds
magnetisation and NdFeB density; ElectroMagnet adds current, turns, mu_r
and a copper-plus-core density. Geometry and material come from
baseline_code.py's classes of the same names.
"""

import numpy as np
from config import DENSITY_COPPER, DENSITY_NDFEB, MU_0


class Magnet:
    """
    A cylindrical magnet, given by its geometry and orientation. The axis
    is `angle` (a unit vector, +Z by default) and `position` is the centre.
    """

    def __init__(self, height, radius, position):
        self.height   = float(height)
        self.radius   = float(radius)
        self.position = np.asarray(position, dtype=float)
        self.angle    = np.array([0.0, 0.0, 1.0])  # cylinder axis, unit vector

        self.volume = self.height * np.pi * self.radius ** 2
        self.mass   = 0.0  # set by subclasses

    def contains(self, points):
        """
        Boolean mask over an (N, 3) array, True where a point is inside
        this cylinder at its current orientation (self.angle).
        """
        points = np.atleast_2d(points)

        axis = np.asarray(self.angle, dtype=float)
        axis = axis / np.linalg.norm(axis)

        rel      = points - self.position
        z_local  = rel @ axis
        radial   = rel - np.outer(z_local, axis)
        r        = np.linalg.norm(radial, axis=1)

        return (np.abs(z_local) <= self.height / 2) & (r <= self.radius)


class PermMagnet(Magnet):
    """
    Permanent NdFeB magnet. magnetic_strength is in A/m and is stored
    divided by mu_0, to match the convention in B_calc.
    """

    def __init__(self, height, radius, position, magnetic_strength):
        super().__init__(height, radius, position)

        self.magnetic_strength = float(magnetic_strength) / MU_0
        self.mass = DENSITY_NDFEB * self.volume


class ElectroMagnet(Magnet):
    """
    Electromagnet with a soft-iron core. Extra fields: current (A), turns,
    mu_r (core relative permeability, default 3000), density_core
    (kg/m^3, default 4800). Mass is copper winding plus core.
    """

    def __init__(self, height, radius, position,
                 current, turns,
                 mu_r=3000.0, density_core=4800.0):
        super().__init__(height, radius, position)

        self.current      = float(current)
        self.turns        = int(turns)
        self.mu_r         = float(mu_r)
        self.density_core = float(density_core)

        self.mass = (DENSITY_COPPER + density_core) * self.volume
