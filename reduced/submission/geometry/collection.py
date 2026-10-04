"""
Collection treats a set of Magnet objects as one rigid body: centre of
mass, total mass and volume, ring grouping for current assignment,
rigid-body rotation (ChangeAngle) and the inertia tensor. Ported from
baseline_code.py; compute_body_inertia_tensor is new for the quaternion
attitude path.
"""

import numpy as np

from geometry.shapes import Magnet
from geometry.transforms import rotation_matrix_from_euler


class Collection:
    """A rigid assembly of Magnet objects. `magnets` must be non-empty."""

    def __init__(self, magnets):
        self.Magnets = list(magnets)

        self.TotalMass   = sum(m.mass   for m in self.Magnets)
        self.TotalVolume = sum(m.volume for m in self.Magnets)

        # mass-weighted mean position
        self.Position = (
            sum(m.position * m.mass for m in self.Magnets) / self.TotalMass
        )

        # cumulative rotation, updated by ChangeAngle
        self.Roll  = 0.0
        self.Pitch = 0.0
        self.Yaw   = 0.0

        # ring grouping: MagnetRings[i] is the list of indices into
        # self.Magnets for ring i, xRings[i] its normalised radius
        self.MagnetRings = []
        self.xRings      = []
        self._build_rings()

        # body-frame inertia tensor, cached now (see
        # compute_body_inertia_tensor for why it can't be rebuilt from
        # positions later)
        self._I_body = self._build_body_inertia_tensor()

    # Ring grouping

    def _build_rings(self):
        # added by Wilf
        """
        Group magnets into rings of equal normalised radius
        sqrt((x/x_max)^2 + (y/y_max)^2) / sqrt(2), the original's
        convention. Uses np.isclose, not ==, to match radii (a real
        ring-building bug in baseline_code.py).
        """
        xs = [m.position[0] for m in self.Magnets]
        ys = [m.position[1] for m in self.Magnets]
        x_max = max(xs) if max(xs) != 0 else 1.0
        y_max = max(ys) if max(ys) != 0 else 1.0

        for i, magnet in enumerate(self.Magnets):
            r = (
                np.sqrt(
                    (magnet.position[0] / x_max) ** 2
                    + (magnet.position[1] / y_max) ** 2
                )
                / np.sqrt(2)
            )

            # attach to an existing ring at the same radius, else start one
            matched = False
            for j, existing_r in enumerate(self.xRings):
                if np.isclose(r, existing_r):
                    self.MagnetRings[j].append(i)
                    matched = True
                    break

            if not matched:
                self.xRings.append(r)
                self.MagnetRings.append([i])

    # Rigid-body rotation

    def ChangeAngle(self, roll, pitch, yaw):
        """
        Rotate the whole assembly in place about its COM (Euler angles in
        radians). Updates every magnet's position and axis, and records
        the angles so compute_inertia_tensor sees the current orientation.
        """
        R = rotation_matrix_from_euler(roll, pitch, yaw)

        for magnet in self.Magnets:
            magnet.angle    = R @ np.array([0.0, 0.0, 1.0])
            rel             = magnet.position - self.Position
            magnet.position = self.Position + R @ rel

        self.Roll  = roll
        self.Pitch = pitch
        self.Yaw   = yaw

    # Inertia tensor

    def compute_inertia_tensor(self):
        """
        3x3 inertia tensor about the assembly COM. Each magnet is a solid
        cylinder; the parallel-axis theorem shifts each local tensor to
        the COM.
        """
        I_total = np.zeros((3, 3))
        COM     = self.Position
        R       = rotation_matrix_from_euler(self.Roll, self.Pitch, self.Yaw)

        for magnet in self.Magnets:
            mass = magnet.mass
            r    = magnet.radius
            h    = magnet.height

            # Local inertia of a solid cylinder about its own COM
            Ixx = (1.0 / 12.0) * mass * (3 * r ** 2 + h ** 2)
            Iyy = Ixx
            Izz = 0.5 * mass * r ** 2
            I_local = np.diag([Ixx, Iyy, Izz])

            # Rotate into world frame
            I_world = R @ I_local @ R.T

            # Parallel-axis shift to assembly COM
            d       = magnet.position - COM
            I_shift = mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

            I_total += I_world + I_shift

        return I_total

    def _build_body_inertia_tensor(self):
        # added by Wilf
        """
        Inertia tensor about the COM in the body frame, from the magnet
        geometry at construction time with no Roll/Pitch/Yaw applied.
        Not recomputed after ChangeAngle: Euler's equation wants I
        constant in the body frame, and rebuilding it from rotated
        geometry would make it orientation-dependent. The SO(3) path
        holds this fixed for integration.
        """
        I_total = np.zeros((3, 3))
        COM     = self.Position

        for magnet in self.Magnets:
            mass = magnet.mass
            r    = magnet.radius
            h    = magnet.height

            Ixx = (1.0 / 12.0) * mass * (3 * r ** 2 + h ** 2)
            Iyy = Ixx
            Izz = 0.5 * mass * r ** 2
            I_local = np.diag([Ixx, Iyy, Izz])

            d       = magnet.position - COM
            I_shift = mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

            I_total += I_local + I_shift

        return I_total

    def compute_body_inertia_tensor(self):
        # added by Wilf
        """
        The body-frame inertia tensor cached at construction, as a
        mutable copy. Use this rather than the world-frame
        compute_inertia_tensor wherever I has to stay valid as the
        assembly rotates during integration (the SO(3) path).
        """
        return self._I_body.copy()
