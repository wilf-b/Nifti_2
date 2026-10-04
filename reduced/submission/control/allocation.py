"""
Turns a desired Wrench into a per-coil current correction, the way
thruster allocation does: the controller asks for a force/torque, this
works out how to get it from the coils. assign_currents_for_pod
(track/control.py) still sets the baseline levitation current; the
allocator only adds a correction on top.

group_by="magnet" drives one current per coil (this is nigels specified driving method. 
group_by="row" shares a current per TrackRow and Fy/roll at y=0. pod_model="point" is a single COM dipole and can't reach
yaw at level attitude; pod_model="multi" can. The two models scale their
moments differently, so the realised force/torque isn't comparable between them without rescaling.

B is exactly proportional to coil current, so the Jacobians are exact.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from geometry.collection import Collection
from geometry.shapes import ElectroMagnet
from physics.field_query import field_gradient_at_point, field_at_points
from control.base import Wrench
from physics.dipole import multi_dipole_force_torque


def _sum_field(magnets, pt):
    # added by Wilf
    B = np.zeros(3)
    for mag in magnets:
        B += field_at_points(mag, pt)[0]
    return B


def _sum_gradient(magnets, pt):
    # added by Wilf
    grad = np.zeros((3, 3))
    for mag in magnets:
        grad += field_gradient_at_point(mag, pt)
    return grad


@dataclass
class ActuatorGroup:
    # added by Wilf
    """Electromagnets driven off one current source. magnet_indices is a
    list even for a single coil."""
    group_id: int
    magnet_indices: list


def group_electromagnets_by_magnet(track):
    # added by Wilf
    """One group per coil; permanent magnets are skipped."""
    return [ActuatorGroup(group_id=i, magnet_indices=[i])
            for i, magnet in enumerate(track.Magnets)
            if isinstance(magnet, ElectroMagnet)]


def group_electromagnets_by_row(track):
    # added by Wilf
    """One group per TrackRow. The magnets must carry a row_id, i.e. have
    been built via track_row_to_magnets()."""
    rows = {}
    for i, magnet in enumerate(track.Magnets):
        if not isinstance(magnet, ElectroMagnet):
            continue
        row_id = getattr(magnet, "row_id", None)
        if row_id is None:
            raise ValueError(f"Magnet {i} has no row_id -- build the track "
                             "via track_row_to_magnets() to use group_by='row'.")
        rows.setdefault(row_id, []).append(i)

    return [ActuatorGroup(group_id=r, magnet_indices=idxs)
            for r, idxs in sorted(rows.items())]


def build_effectiveness_jacobian(track, groups, pod_position, m_vec, h=1e-5):
    # added by Wilf
    """Column j is the force (Fx, Fy, Fz) on the pod dipole per amp in
    group j, from Force = m_vec . grad(B)."""
    J = np.zeros((3, len(groups)))
    for col, group in enumerate(groups):
        for idx in group.magnet_indices:
            magnet = track.Magnets[idx]
            original_current = magnet.current
            magnet.current = 1.0
            grad = field_gradient_at_point(magnet, pod_position, h=h)
            magnet.current = original_current
            J[:, col] += m_vec @ grad
    return J


def build_torque_jacobian(track, groups, pod_position, m_vec):
    # added by Wilf
    """Column j is the world-frame torque (Tx, Ty, Tz) on the pod dipole
    per amp in group j."""
    J = np.zeros((3, len(groups)))
    for col, group in enumerate(groups):
        B_group = np.zeros(3)
        for idx in group.magnet_indices:
            magnet = track.Magnets[idx]
            original_current = magnet.current
            magnet.current = 1.0
            B_group += field_at_points(magnet, pod_position)[0]
            magnet.current = original_current
        J[:, col] = np.cross(m_vec, B_group)
    return J


def build_multi_dipole_wrench_jacobian(track, groups, pod, R, COM_world):
    # added by Wilf
    """Combined (6, n_groups) [F; tau] Jacobian for pod_model="multi":
    column j is the wrench from 1 A in group j's magnets, carrying the
    per-magnet lever-arm term that makes yaw reachable. R is the
    world-from-body rotation, COM_world the pod COM."""
    J = np.zeros((6, len(groups)))
    for col, group in enumerate(groups):
        group_magnets = [track.Magnets[idx] for idx in group.magnet_indices]
        originals = [m.current for m in group_magnets]
        for m in group_magnets:
            m.current = 1.0

        def B_fn(pt):
            return _sum_field(group_magnets, pt)

        def gradB_fn(pt):
            return _sum_gradient(group_magnets, pt)

        F, T = multi_dipole_force_torque(pod, R, COM_world, B_fn, gradB_fn)
        J[:3, col] = F
        J[3:, col] = T

        for m, c in zip(group_magnets, originals):
            m.current = c
    return J


class CurrentAllocator:
    # added by Wilf
    """
    Weighted least-squares fit from a desired Wrench to a per-coil current
    correction, added on top of the assign_currents_for_pod baseline. Each
    coil is clipped to [0, current_limit] (the rig can't run current
    backwards). Fz is always in the fit, weighted by fz_weight; torque
    enters only when include_torque is True.

    torque_weight/tz_weight sit above fz_weight because force and torque
    share one fit and reachable torque is much smaller in raw units here.
    None of the weights or gains in this repo are rig-calibrated.
    """

    def __init__(self, current_limit, fz_weight=100.0, rcond=1e-3,
                 include_torque=False,
                 torque_weight=1000.0, tz_weight=100000.0,
                 solver="pinv_clip", pod_model="point", group_by="magnet"):
        self.current_limit = current_limit
        self.fz_weight = fz_weight
        self.rcond = rcond
        self.include_torque = include_torque
        self.torque_weight = torque_weight
        self.tz_weight = tz_weight
        self.solver = solver
        self.pod_model = pod_model
        self.group_by = group_by

        # last allocate() call's intermediates, for scripts that inspect
        # realised vs commanded force/torque
        self.last_dI = np.zeros(0)
        self.last_realized_force = np.zeros(3)
        self.last_realized_torque = None

    def allocate(self, wrench, track, pod_position, m_vec=None, *,
                 pod=None, R=None):
        """
        Add a per-group current correction to track.Magnets[i].current in
        place, clipped per group so magnets on one source stay equal.
        pod_model="point" uses m_vec; pod_model="multi" uses pod and R.
        """
        groups = (group_electromagnets_by_row(track) if self.group_by == "row"
                  else group_electromagnets_by_magnet(track))
        if not groups:
            return

        if self.pod_model == "multi":
            if pod is None or R is None:
                raise ValueError("pod_model='multi' needs pod and R, not m_vec.")
            J6 = build_multi_dipole_wrench_jacobian(track, groups, pod, R, pod_position)
            J_force = J6[:3, :]
            J_torque = J6[3:, :]
        else:
            if m_vec is None:
                raise ValueError("pod_model='point' needs m_vec.")
            J_force = build_effectiveness_jacobian(track, groups, pod_position, m_vec)
            J_torque = (build_torque_jacobian(track, groups, pod_position, m_vec)
                        if self.include_torque else None)

        if self.include_torque:
            J = np.vstack([J_force, J_torque])
            if self.pod_model == "multi":
                # Tz is reachable under this model, so target it like Tx/Ty
                target = np.array([wrench.force[0], wrench.force[1], wrench.force[2],
                                   wrench.torque[0], wrench.torque[1], wrench.torque[2]])
                weights = np.array([1.0, 1.0, self.fz_weight,
                                    self.torque_weight, self.torque_weight, self.torque_weight])
            else:
                # Tz is unreachable under the point model, so pin it to 0
                target = np.array([wrench.force[0], wrench.force[1], wrench.force[2],
                                   wrench.torque[0], wrench.torque[1], 0.0])
                weights = np.array([1.0, 1.0, self.fz_weight,
                                    self.torque_weight, self.torque_weight, self.tz_weight])
        else:
            J = J_force
            target = np.array([wrench.force[0], wrench.force[1], wrench.force[2]])
            weights = np.array([1.0, 1.0, self.fz_weight])

        J_weighted = weights[:, None] * J
        target_weighted = weights * target

        # tightest per-group dI range that keeps every magnet in [0, current_limit]
        group_bounds = []
        for group in groups:
            group_currents = np.array(
                [track.Magnets[idx].current for idx in group.magnet_indices]
            )
            lo = float((0.0 - group_currents).max())
            hi = float((self.current_limit - group_currents).min())
            if hi < 0.0:
                # dI=0 must stay feasible; hi<0 means the baseline already
                # exceeds current_limit (a stale limit vs a rescaled baseline)
                raise ValueError(
                    f"current_limit={self.current_limit} is below group "
                    f"{group.group_id}'s baseline current "
                    f"({group_currents.max():.4f} A)."
                )
            group_bounds.append((min(lo, hi), max(lo, hi)))

        if self.solver == "bounded":
            lb = np.array([lo for lo, _ in group_bounds])
            ub = np.array([hi for _, hi in group_bounds])
            dI = lsq_linear(J_weighted, target_weighted, bounds=(lb, ub)).x
        else:
            dI_unclipped = np.linalg.pinv(J_weighted, rcond=self.rcond) @ target_weighted
            dI = np.zeros(len(group_bounds))
            for j, (lo, hi) in enumerate(group_bounds):
                dI[j] = np.clip(dI_unclipped[j], lo, hi)

        self.last_dI = dI.copy()
        self.last_realized_force = J_force @ dI
        self.last_realized_torque = (J_torque @ dI) if self.include_torque else None

        for group, delta in zip(groups, dI):
            for idx in group.magnet_indices:
                track.Magnets[idx].current += float(delta)
