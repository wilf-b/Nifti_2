"""
The empirical current-assignment formula: each coil's current from the
pod's radial XY position, the coil's ring, and scaling for magnet count
and pod mass. baseline_code.py inlined this in both
energy_potential_controlled and energy_potential_static; it's one place
now.
"""

import numpy as np

from geometry.collection import Collection
from config import SimConfig, DEFAULT_SIM_CONFIG


def assign_currents(track, pod_xy, config=DEFAULT_SIM_CONFIG):
    # added by Wilf
    """
    Set .current on every ElectroMagnet in the track, in place, before
    the field is computed at each pod position. Each ring's current comes
    from an empirical quadratic fit to the rig; coils in a ring share it.

    pod_xy is the pod's radial distance from track centre, np.hypot(x, y).
    config supplies reference_magnet_count, reference_pod_mass and
    z_proportion. For ring i with normalised radius xRings[i]:

        scale   = z_proportion * mass_proportion * magnet_proportion
        arg     = xRings[i] * 2.5 + 1221.08 * pod_xy
        current = scale * (1.8 + 0.032 * arg^2 + 39.34 * pod_xy) / 1000
    """
    magnet_proportion = config.reference_magnet_count / len(track.Magnets)
    # TotalMass-equivalent, as the original had it (Pod.TotalMass/0.1459);
    # assign_currents_for_pod uses the real pod mass instead
    mass_proportion   = track.Magnets[0].mass * len(track.Magnets) / config.reference_pod_mass
    z_proportion      = config.z_proportion

    scale = z_proportion * magnet_proportion

    for ring_i, ring in enumerate(track.MagnetRings):
        arg     = track.xRings[ring_i] * 2.5 + 1221.08 * pod_xy
        current = scale * (1.8 + 0.032 * arg ** 2 + 39.34 * pod_xy) / 1000

        for magnet_i in ring:
            track.Magnets[magnet_i].current = current


def assign_currents_for_pod(track, pod_xy, pod_total_mass,
                            config=DEFAULT_SIM_CONFIG, current_scale=1.0):
    # added by Wilf
    """
    Current assignment scaled by the pod's real mass; the version
    simulation uses. pod_xy is np.hypot(x, y), pod_total_mass is
    pod.TotalMass in kg.

    current_scale (default 1.0) is a uniform multiplier on top of the
    formula's own scale. The formula's proportion normalisation doesn't
    capture actual lift-per-amp for a given track/pod (~60x apart across
    two presets), so calibrate_levitation solves current_scale per
    track/pod rather than hardcoding it.
    """
    magnet_proportion = config.reference_magnet_count / len(track.Magnets)
    mass_proportion   = pod_total_mass / config.reference_pod_mass
    z_proportion      = config.z_proportion

    scale = z_proportion * mass_proportion * magnet_proportion * current_scale

    for ring_i, ring in enumerate(track.MagnetRings):
        arg     = track.xRings[ring_i] * 2.5 + 1221.08 * pod_xy
        current = scale * (1.8 + 0.032 * arg ** 2 + 39.34 * pod_xy) / 1000

        for magnet_i in ring:
            track.Magnets[magnet_i].current = current
