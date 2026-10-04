"""
Analytic ideal fields: the theoretical target fields used by
energy_potential_ideal and Ideal_Check. They don't depend on any real
magnet geometry, they just say what the field should look like for stable
levitation.

Ported from baseline_code.py's B_field_Valley and B_field_Gravity. The two
tuning constants below were hardcoded in the originals; they're named
here so they can be changed without touching the sweep logic.
"""

import numpy as np

from config import G


VALLEY_SCALE    = 5.0    # overall field scale for the restoring valley
VALLEY_Z_CENTRE = 0.40   # z-height of the stability centre (m)


def B_field_valley(point):
    """
    Field (Tesla) built to give a restoring lateral force: a bowl in x-y
    centred at z = VALLEY_Z_CENTRE, so a displaced pod is pushed back
    toward (0, 0, VALLEY_Z_CENTRE). `point` is (x, y, z) in metres.
    """
    x, y, z = point
    dz = z - VALLEY_Z_CENTRE
    return VALLEY_SCALE * np.array([
        -x * dz,
        -y * dz,
         x**2 + y**2 + dz**2,
    ])


def B_field_gravity(point, total_mass, n_points):
    """
    Field (Tesla) that produces a force balancing gravity on the pod.
    Proportional to mass per point, so summed over all n_points pod
    sample points the upward force comes to m*g. `point` is (x, y, z) in
    metres, total_mass in kg.
    """
    x, y, z = point
    M = total_mass / n_points
    return -np.array([
        -0.5 * M * G * x,
        -0.5 * M * G * y,
              M * G * z,
    ])
