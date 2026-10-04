"""
Levitation current-scale calibration.

assign_currents_for_pod's schedule has roughly the right shape but is
scaled far too small to levitate anything tested, by a different factor
per preset. Force is linear in current, so one field evaluation at the
schedule's peak-lift height gives the multiplier that matches the pod's
weight. That peak is only half-stable, so calibrate_levitation adds a
margin and returns z_stable, a point just past the peak where Fz crosses
the weight with restoring slope (not a true equilibrium - Earnshaw -
just a better operating point).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar, brentq

from geometry.collection import Collection
from physics.field_query import field_gradient_at_point
from physics.dipole import dipole_moment
from track.control import assign_currents_for_pod
from config import SimConfig, DEFAULT_SIM_CONFIG
from config import G


@dataclass
class LevitationCalibration:
    # added by Wilf
    """
    z_ref is the peak-lift height (m) at pod_xy=0. z_stable, above z_ref
    where Fz = weight with dFz/dz < 0, is the height to actually operate
    at. current_scale feeds assign_currents_for_pod's current_scale.
    Fz_at_peak_unscaled is the raw lift at z_ref and margin the safety
    factor used.
    """
    z_ref: float
    z_stable: float
    current_scale: float
    Fz_at_peak_unscaled: float
    margin: float


def calibrate_levitation(track, pod, pod_pts, config=DEFAULT_SIM_CONFIG,
                         pod_xy=0.0, z_bounds=None, n_coarse=200,
                         singularity_margin_factor=4.0, margin=1.2):
    # added by Wilf
    """
    Find the height and current scale that make this track/pod's schedule
    lift its own weight, at the given pod_xy (default track centre) and
    level attitude. Mutates track.Magnets[i].current, so re-run
    assign_currents_for_pod with the returned current_scale afterwards.

    pod_xy matters a lot: the schedule swings sharply with it (the
    1221.08*pod_xy term), so a calibration at pod_xy=0 is orders of
    magnitude off by pod_xy=0.005. Calibrate near where the scenario
    runs. Attitude always assumes level; a pre-tilted start really does
    see less Fz, which is for the controller to correct.

    z_bounds defaults to (singularity_margin_factor * tallest coil
    half-height, 0.30); the lower bound keeps the search off the field
    model's near-edge singularity. Raises ValueError if no height lifts
    at all, if the peak sits within 10% of a coil half-height, or if no
    z_stable crossing is found (widen z_bounds[1]).
    """
    if z_bounds is None:
        max_half_height = max(magnet.height for magnet in track.Magnets) / 2
        z_bounds = (singularity_margin_factor * max_half_height, 0.30)

    m_body = dipole_moment(pod, pod_pts)
    assign_currents_for_pod(track, pod_xy=pod_xy, pod_total_mass=pod.TotalMass, config=config)

    def Fz_at(z):
        pos = np.array([0.0, pod_xy, z])
        grad = np.zeros((3, 3))
        for magnet in track.Magnets:
            grad += field_gradient_at_point(magnet, pos)
        return float((m_body @ grad)[2])

    zs = np.linspace(z_bounds[0], z_bounds[1], n_coarse)
    Fz_coarse = np.array([Fz_at(z) for z in zs])
    idx_peak = int(np.argmax(Fz_coarse))

    # refine the coarse peak: bounded search over the two samples either side
    lo = zs[max(0, idx_peak - 2)]
    hi = zs[min(len(zs) - 1, idx_peak + 2)]
    result = minimize_scalar(lambda z: -Fz_at(z), bounds=(lo, hi), method="bounded")

    z_ref = float(result.x)
    Fz_ref = -float(result.fun)

    if Fz_ref <= 0:
        raise ValueError(
            f"No height in {z_bounds} produces positive lift for this "
            "track/pod; the schedule's shape can't levitate it at any scale."
        )

    for magnet in track.Magnets:
        half_height = magnet.height / 2
        if abs(z_ref - half_height) < 0.1 * half_height:
            raise ValueError(
                f"Peak at z={z_ref:.5f} sits within 10% of a coil half-height "
                f"({half_height:.5f}), likely the field model's near-edge "
                "singularity. Increase z_bounds[0] or singularity_margin_factor."
            )

    target_weight = pod.TotalMass * G
    current_scale = (margin * target_weight) / Fz_ref

    # z_stable: where Fz_scaled crosses the weight past z_ref, on the
    # descending branch
    def Fz_scaled_minus_target(z):
        return current_scale * Fz_at(z) - target_weight

    hi_search = z_bounds[1]
    if Fz_scaled_minus_target(hi_search) > 0:
        raise ValueError(
            f"No z_stable found: Fz stays above the weight all the way to "
            f"z_bounds[1]={hi_search}. Widen z_bounds."
        )
    z_stable = float(brentq(Fz_scaled_minus_target, z_ref, hi_search))

    return LevitationCalibration(z_ref=z_ref, z_stable=z_stable,
                                 current_scale=current_scale,
                                 Fz_at_peak_unscaled=Fz_ref, margin=margin)


def local_current_scale(track, pod_position, m_world, pod_total_mass,
                        config=DEFAULT_SIM_CONFIG, margin=1.0):
    # added by Wilf
    """
    The current_scale making the unscaled schedule's Fz at this live pod
    position and orientation equal margin * weight. calibrate_levitation's
    single scale goes badly stale a few mm away because the baseline is a
    steep function of pod_xy; this re-solves it live (one field
    evaluation), so it's meant to be the callable current_scale passed to
    run_closed_loop_dynamics_so3. It does not re-solve z_ref/z_stable.
    m_world is the rotated dipole moment. Mutates track.Magnets[i].current
    and raises ValueError if Fz isn't positive here.
    """
    pod_xy = float(np.hypot(pod_position[0], pod_position[1]))
    assign_currents_for_pod(track, pod_xy=pod_xy, pod_total_mass=pod_total_mass, config=config)

    grad = np.zeros((3, 3))
    for magnet in track.Magnets:
        grad += field_gradient_at_point(magnet, pod_position)
    Fz_unscaled = float((m_world @ grad)[2])

    if Fz_unscaled <= 0:
        raise ValueError(
            f"No positive lift at pod_position={pod_position} under the "
            "unscaled schedule; no current_scale fixes that."
        )

    target_weight = pod_total_mass * G
    return (margin * target_weight) / Fz_unscaled


def calibrate_for_target(track, pod, pod_pts, config, target_pod_xy,
                         current_limit_margin=5.0):
    # added by Wilf
    """
    calibrate_levitation, then assign_currents_for_pod at that scale,
    then current_limit = margin * baseline max current. Deriving
    current_limit live matters: a hardcoded one can sit below the
    preset-dependent baseline and make CurrentAllocator.allocate raise.
    Returns (cal, current_limit).
    """
    cal = calibrate_levitation(track, pod, pod_pts, config, pod_xy=target_pod_xy)
    assign_currents_for_pod(track, pod_xy=target_pod_xy, pod_total_mass=pod.TotalMass,
                            config=config, current_scale=cal.current_scale)
    current_limit = current_limit_margin * max(m.current for m in track.Magnets)
    return cal, current_limit


def make_live_scale_fn(track, pod, config, allocator, initial_scale,
                       current_limit_margin=5.0, margin=1.0):
    # added by Wilf
    """
    Per-step current_scale closure for run_closed_loop_dynamics_so3.
    Re-solves local_current_scale at the pod's live position, falls back
    to the last good scale on ValueError (a transiently unliftable pose),
    and keeps allocator.current_limit in step with the live baseline.
    """
    last_scale = initial_scale

    def scale_fn(pos, m_world):
        nonlocal last_scale
        try:
            scale = local_current_scale(track, pos, m_world, pod.TotalMass, config, margin=margin)
        except ValueError:
            scale = last_scale
            assign_currents_for_pod(track, pod_xy=float(np.hypot(pos[0], pos[1])),
                                    pod_total_mass=pod.TotalMass, config=config)
        allocator.current_limit = current_limit_margin * max(m.current for m in track.Magnets) * scale
        last_scale = scale
        return scale

    return scale_fn
