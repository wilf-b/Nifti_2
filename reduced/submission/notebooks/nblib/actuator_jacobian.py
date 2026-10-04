"""
Support code for notebook 02 (actuator-authority limits), pulled out so the
notebook reads as narrative. It loads a fixed Classic Track / Two Stacks
scene, calibrates the baseline levitation currents, and probes the real
CurrentAllocator and the raw effectiveness/torque Jacobians.

The point model's dipole_moment is TotalVolume/N smaller than the per-magnet
volume the multi-dipole Jacobian uses, so the multi path's moments are
divided by N (`scaled_moments`) before allocating -- the same reconciliation
tests/test_control.py's pod_model="multi" section describes.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np

from data_io.presets import load_track_preset, load_pod_preset
from geometry.sampling import generate_pod_base_points
from geometry.transforms import rotation_matrix_from_euler
from physics.dipole import dipole_moment
from control.allocation import (
    CurrentAllocator, group_electromagnets_by_magnet, group_electromagnets_by_row,
    build_effectiveness_jacobian, build_torque_jacobian,
)
from control.base import Wrench
from simulation.calibration import calibrate_levitation
from track.control import assign_currents_for_pod
from config import SimConfig

TRACK_NAME, POD_NAME = "Classic Track", "Two Stacks"
POD_POS = np.array([0.0, 0.0, 0.040])
R_LEVEL = np.eye(3)
FORCE_TARGET, TORQUE_TARGET = 0.1, 0.01
CURRENT_LIMIT_FACTOR = 5.0
DOF_LABELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
TARGETS = np.array([FORCE_TARGET] * 3 + [TORQUE_TARGET] * 3)

# point-model solve weights: down-weight force, up-weight torque, and (unlike
# the allocator) target Tz like Tx/Ty so the rank deficiency shows up as a
# small realised fraction rather than a hard zero.
POINT_W = np.array([1.0, 1.0, 100.0, 1000.0, 1000.0, 1000.0])
POINT_RCOND = 1e-3


@dataclass
class Scene:
    track: object
    pod: object
    pod_pts: np.ndarray
    n_points: int
    baseline_currents: list
    current_limit: float


@contextmanager
def scaled_moments(pod, factor):
    """Temporarily divide every magnet's magnetic_strength by `factor`."""
    original = [m.magnetic_strength for m in pod.Magnets]
    try:
        for m in pod.Magnets:
            m.magnetic_strength /= factor
        yield
    finally:
        for m, s in zip(pod.Magnets, original):
            m.magnetic_strength = s


def _reset_currents(scene):
    for m, base in zip(scene.track.Magnets, scene.baseline_currents):
        m.current = base


def load_scene():
    """Classic Track / Two Stacks with calibrated baseline currents."""
    track, _ = load_track_preset(TRACK_NAME)
    pod = load_pod_preset(POD_NAME)
    pod_pts = generate_pod_base_points(pod)
    n_points = pod_pts.shape[0]

    cal = calibrate_levitation(track, pod, pod_pts, SimConfig(), pod_xy=0.0)
    assign_currents_for_pod(track, pod_xy=0.0, pod_total_mass=pod.TotalMass,
                            config=SimConfig(), current_scale=cal.current_scale)
    current_limit = CURRENT_LIMIT_FACTOR * max(m.current for m in track.Magnets)
    return Scene(track, pod, pod_pts, n_points,
                 [m.current for m in track.Magnets], current_limit)


def six_dof_authority(scene):
    """One unit wrench per DOF through CurrentAllocator(pod_model="multi"),
    dead level / on-centre. Returns (ratios, max |dI| per DOF, realized wrenches)."""
    ratios, currents, realized = [], [], []
    with scaled_moments(scene.pod, scene.n_points):
        for i in range(6):
            _reset_currents(scene)
            force, torque = np.zeros(3), np.zeros(3)
            if i < 3:
                force[i] = FORCE_TARGET
            else:
                torque[i - 3] = TORQUE_TARGET
            alloc = CurrentAllocator(current_limit=scene.current_limit, fz_weight=100.0,
                                     include_torque=True, torque_weight=1000.0, pod_model="multi")
            alloc.allocate(Wrench(force=force, torque=torque), scene.track, POD_POS,
                           pod=scene.pod, R=R_LEVEL)
            got = alloc.last_realized_force[i] if i < 3 else alloc.last_realized_torque[i - 3]
            tgt = FORCE_TARGET if i < 3 else TORQUE_TARGET
            ratios.append(min(got / tgt, 1.0))
            currents.append(float(np.max(np.abs(alloc.last_dI))))
            realized.append(np.concatenate([alloc.last_realized_force, alloc.last_realized_torque]))
    return np.array(ratios), np.array(currents), np.array(realized)


def point_cross_coupling(scene, rpy):
    """6x6 realised/commanded matrix under pod_model="point" at attitude `rpy`,
    from the weighted-pinv solve. Returns (M, max |dI| per DOF, singular values of J6)."""
    m_world = rotation_matrix_from_euler(*rpy) @ dipole_moment(scene.pod, scene.pod_pts)
    _reset_currents(scene)
    groups = group_electromagnets_by_magnet(scene.track)
    J6 = np.vstack([build_effectiveness_jacobian(scene.track, groups, POD_POS, m_world),
                    build_torque_jacobian(scene.track, groups, POD_POS, m_world)])
    pinv = np.linalg.pinv(POINT_W[:, None] * J6, rcond=POINT_RCOND)

    realised, di_max = [], []
    for i in range(6):
        tgt = np.zeros(6)
        tgt[i] = FORCE_TARGET if i < 3 else TORQUE_TARGET
        dI = pinv @ (POINT_W * tgt)
        realised.append(J6 @ dI)
        di_max.append(float(np.abs(dI).max()))
    return (np.array(realised) / TARGETS[:, None], np.array(di_max),
            np.linalg.svd(J6, compute_uv=False))


def row_jacobian_svd():
    """Classic Track's per-row stacked 6-target Jacobian at the tilted pose --
    the sigma_4 authority ceiling for section 3. Self-contained (own pose)."""
    pose = np.array([0.0, 0.005, 0.040])
    track, _ = load_track_preset("Classic Track")
    pod = load_pod_preset(POD_NAME)
    m_world = rotation_matrix_from_euler(0.15, -0.10, 0.20) @ dipole_moment(
        pod, generate_pod_base_points(pod))
    groups = group_electromagnets_by_row(track)
    J6 = np.vstack([build_effectiveness_jacobian(track, groups, pose, m_world),
                    build_torque_jacobian(track, groups, pose, m_world)])
    return (np.linalg.svd(J6, compute_uv=False),
            np.linalg.matrix_rank(J6, tol=1e-9), len(groups))
