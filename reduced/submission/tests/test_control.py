"""
control/ tests: allocation and the PD controllers. Plain top-to-bottom
asserts, run with `python3 tests/test_control.py`. Most sections guard a
specific bug found during the control work. Needs data/Presets/ for the
pod_model section.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import (
    synthetic_electro_track, realized_force, realized_torque,
    load_classic_two_stacks, TILT_R, TILTED_POSE, POD_POS, M_VEC_TEST,
)

import numpy as np
from scipy.integrate import solve_ivp

from geometry.transforms import rotation_matrix_from_euler
from geometry.quaternion import quat_identity, quat_to_matrix, matrix_to_quat, quat_normalize, quat_derivative, vee
from physics.field_query import field_gradient_at_point
from track.control import assign_currents_for_pod
from data_io.presets import load_track_preset
from simulation.dynamics import DynamicsState, DynamicsStateSO3
from simulation.calibration import calibrate_levitation
from config import SimConfig

from control.allocation import (
    group_electromagnets_by_magnet, group_electromagnets_by_row,
    build_effectiveness_jacobian, build_torque_jacobian,
    build_multi_dipole_wrench_jacobian, CurrentAllocator,
)
from control.base import Wrench
from control.xy_controller import LateralTarget, LateralPDController
from control.z_controller import ZTarget, ZPDController
from control.attitude_controller import AttitudeTarget, GeometricAttitudePDController
from control.combined_controller import CombinedController

track = synthetic_electro_track()   # 3x3 electro grid, all coils at 0.05 A
groups = group_electromagnets_by_magnet(track)

# effectiveness Jacobian; field is exactly linear in current
J = build_effectiveness_jacobian(track, groups, POD_POS, M_VEC_TEST)
assert J.shape == (3, 9) and np.all(np.isfinite(J))

# double one coil's current, expect double the field. the allocator leans hard
# on this being exact, so rtol is tight. put the current back afterwards.
probe = track.Magnets[groups[0].magnet_indices[0]]
probe.current = 1.0
g1 = field_gradient_at_point(probe, POD_POS)
probe.current = 2.0
g2 = field_gradient_at_point(probe, POD_POS)
probe.current = 0.05
assert np.allclose(g2, 2 * g1, rtol=1e-8)#'replace the prev =='

# Fx realised, Fz left alone, clip is per group not per magnet
allocator = CurrentAllocator(current_limit=1.0, fz_weight=100.0)
F0 = realized_force(track, POD_POS, M_VEC_TEST)
target_fx = 0.05 * J[0, 0]   # the Fx that 0.05 A on coil 0 alone would make - small and reachable
allocator.allocate(Wrench(force=np.array([target_fx, 0.0, 0.0])), track, POD_POS, M_VEC_TEST)
new = np.array([m.current for m in track.Magnets])
assert np.all(new >= 0.0) and np.all(new <= 1.0 + 1e-9)
dF = realized_force(track, POD_POS, M_VEC_TEST) - F0
assert np.isclose(dF[0], target_fx, rtol=0.2)
assert abs(dF[2]) < 0.2 * abs(target_fx) + 1e-12   # pure-Fx request perturbed Fz

# wipe currents to a known floor, then ask for 1000x too much and check the
# clip holds every coil inside [0, limit] rather than clipping the group sum
for m in track.Magnets:
    m.current = 0.0
CurrentAllocator(current_limit=0.01, fz_weight=100.0).allocate(
    Wrench(force=np.array([target_fx * 1000, 0.0, 0.0])), track, POD_POS, M_VEC_TEST)
clipped = np.array([m.current for m in track.Magnets])
assert np.all(clipped >= 0.0) and np.all(clipped <= 0.01 + 1e-9)
for m in track.Magnets:
    m.current = 0.05   # reset for the next section

lt = LateralTarget(x=0.01, y=-0.01)
pd = LateralPDController(target=lt, kp=10.0, kd=1.0, force_limit=0.05)
w = pd.compute(0.0, DynamicsState(x0=0.0, y0=0.0), 0.001)
assert np.dot(w.force[:2], [lt.x, lt.y]) > 0   # force points toward target
assert np.isclose(w.force[2], 0.0) and np.allclose(w.torque, 0.0)   # xy ctrl must not command Fz or torque
assert np.allclose(pd.compute(0.0, DynamicsState(x0=lt.x, y0=lt.y), 0.001).force, 0.0)
w_sat = LateralPDController(target=lt, kp=1e6, kd=0.0, force_limit=0.05).compute(
    0.0, DynamicsState(x0=-10.0, y0=-10.0), 0.001)
assert np.linalg.norm(w_sat.force[:2]) <= 0.05 + 1e-9   # force exceeded force_limit

zpd = ZPDController(target=ZTarget(z=0.05), kp=1000.0, kd=10.0, force_limit=0.5)
wz = zpd.compute(0.0, DynamicsState(z0=0.04), 0.001)
assert wz.force[2] > 0 and np.allclose(wz.force[:2], 0.0) and np.allclose(wz.torque, 0.0)
assert np.allclose(zpd.compute(0.0, DynamicsState(z0=0.05), 0.001).force, 0.0)
wz_sat = ZPDController(target=ZTarget(z=0.05), kp=1e6, kd=0.0, force_limit=0.5).compute(
    0.0, DynamicsState(z0=-10.0), 0.001)
assert np.isclose(abs(wz_sat.force[2]), 0.5, atol=1e-9)   # Fz correction exceeded force_limit

# attitude_controller: gains are left for rig ID, so pin structure not
# numbers -- no force, body-Tz identically zero, zero error -> zero
# torque, roll/pitch decays.
_I = np.diag([0.002, 0.005, 0.008])
_I_inv = np.linalg.inv(_I)


def _e_R(s):
    R = quat_to_matrix(quat_normalize(s[6:10]))
    return 0.5 * vee(R - R.T)


ctrl = GeometricAttitudePDController(AttitudeTarget(roll=0.0, pitch=0.0),
                                    k_R=0.05, k_omega=0.01, torque_limit=1.0)
# two off-axis poses; the big yaw in the second is what really exercises the
# "body-z torque channel stays zero" check
for roll_p, pitch_p, yaw_p in [(0.3, -0.2, 0.5), (-0.4, 0.35, -1.0)]:
    q_p = matrix_to_quat(rotation_matrix_from_euler(roll_p, pitch_p, yaw_p))
    w_p = ctrl.compute(0.0, DynamicsStateSO3(q0=q_p, wx0=0.1, wy0=-0.05, wz0=0.2), 0.001)
    assert np.allclose(w_p.force, 0.0, atol=1e-12)   # attitude controller must never command a force
    assert abs((quat_to_matrix(q_p).T @ w_p.torque)[2]) < 1e-9   # body-z torque channel must be zero
assert np.allclose(ctrl.compute(0.0, DynamicsStateSO3(q0=quat_identity()), 0.001).torque, 0.0, atol=1e-9)

q_err0 = matrix_to_quat(rotation_matrix_from_euler(0.3, -0.2, 0.5))
s0 = [0, 0, 0, 0, 0, 0, *q_err0, 0, 0, 0]
conv = GeometricAttitudePDController(AttitudeTarget(roll=0.0, pitch=0.0),
                                    k_R=0.2, k_omega=0.05, torque_limit=1.0)


def _att_rhs(t, s):
    q = quat_normalize(s[6:10])
    ww = s[10:13]
    tau_body = quat_to_matrix(q).T @ conv.compute(
        t, DynamicsStateSO3(q0=q, wx0=ww[0], wy0=ww[1], wz0=ww[2]), 0.001).torque
    return [0, 0, 0, 0, 0, 0, *quat_derivative(q, ww),
            *(_I_inv @ (tau_body - np.cross(ww, _I @ ww)))]


# 12 s is enough for k_R=0.2 to bleed the initial error down to < 5% of its start
sol_att = solve_ivp(_att_rhs, (0, 12.0), s0, max_step=0.005)
assert np.linalg.norm(_e_R(sol_att.y[:, -1])[:2]) < 0.05 * np.linalg.norm(_e_R(np.array(s0))[:2])

# torque Jacobian: Tz exactly zero at y=0, per-group clip with torque on
J_tau_y0 = build_torque_jacobian(track, groups, POD_POS, M_VEC_TEST)
assert np.allclose(J_tau_y0[2, :], 0.0, atol=1e-12)   # Tz should be exactly zero at y=0
assert np.any(np.abs(J_tau_y0[0, :]) > 1e-9) and np.any(np.abs(J_tau_y0[1, :]) > 1e-9)

pos_offy = np.array([0.0, 0.005, 0.03])
J_tau_offy = build_torque_jacobian(track, groups, pos_offy, M_VEC_TEST)
assert np.any(np.abs(J_tau_offy[0, :]) > 1e-9)   # Tx reachable off y=0

for m in track.Magnets:
    m.current = 0.05
# 'bounded' least-squares so the solver can't hand back negative currents
tq = CurrentAllocator(current_limit=1.0, fz_weight=100.0, include_torque=True, solver="bounded")
T0 = realized_torque(track, pos_offy, M_VEC_TEST)
target_tx = 0.05 * J_tau_offy[0, 0]
tq.allocate(Wrench(force=np.zeros(3), torque=np.array([target_tx, 0.0, 0.0])), track, pos_offy, M_VEC_TEST)
new = np.array([m.current for m in track.Magnets])
assert np.all(new >= -1e-9) and np.all(new <= 1.0 + 1e-9)   # currents out of [0, limit] with torque on
dT = realized_torque(track, pos_offy, M_VEC_TEST) - T0
assert np.isclose(dT[0], target_tx, rtol=0.2)   # Tx not realised
assert abs(dT[2]) < 0.2 * abs(target_tx) + 1e-12   # pure-Tx request perturbed Tz
for m in track.Magnets:
    m.current = 0.05

# CombinedController composition doesn't perturb the sub-laws
combo_state = DynamicsStateSO3(x0=0.02, y0=-0.01, z0=0.05,
                               q0=matrix_to_quat(rotation_matrix_from_euler(0.1, -0.05, 0.0)))
lat = LateralPDController(LateralTarget(x=0.0, y=0.0), kp=10.0, kd=1.0, force_limit=1.0)
att = GeometricAttitudePDController(AttitudeTarget(roll=0.0, pitch=0.0), k_R=0.05, k_omega=0.01, torque_limit=1.0)
combined = CombinedController([lat, att])
wc = combined.compute(0.0, combo_state, 0.001)
w_lat = lat.compute(0.0, combo_state, 0.001)
w_att = att.compute(0.0, combo_state, 0.001)
assert np.allclose(wc.force, w_lat.force) and np.allclose(wc.torque, w_att.torque)
assert np.linalg.norm(w_lat.force[:2]) > 1e-6 and np.linalg.norm(w_att.torque[:2]) > 1e-6

# allocate() used to force the Fz target to 0.0 whatever wrench.force[2]
# was, so ZPDController could never correct z drift.
zc_track = synthetic_electro_track()
zc_groups = group_electromagnets_by_magnet(zc_track)
zc_J = build_effectiveness_jacobian(zc_track, zc_groups, POD_POS, M_VEC_TEST)
F0 = realized_force(zc_track, POD_POS, M_VEC_TEST)
target_fz = 0.05 * zc_J[2, 0]
CurrentAllocator(current_limit=1.0, fz_weight=100.0).allocate(
    Wrench(force=np.array([0.0, 0.0, target_fz])), zc_track, POD_POS, M_VEC_TEST)
assert np.isclose((realized_force(zc_track, POD_POS, M_VEC_TEST) - F0)[2], target_fz, rtol=0.2)

# pod_model='point': tau = m x B is perpendicular to m, so torque about
# the pod's own moment axis is unreachable -- the [Fx..Tz] Jacobian is
# exactly rank 5 under tilt, null direction [0,0,0, m_hat].
dof_track, dof_pod, dof_pod_pts, dof_m_body = load_classic_two_stacks()
dof_groups = group_electromagnets_by_magnet(dof_track)
m_world = TILT_R @ dof_m_body
m_hat = m_world / np.linalg.norm(m_world)

J6 = np.vstack([build_effectiveness_jacobian(dof_track, dof_groups, TILTED_POSE, m_world),
                build_torque_jacobian(dof_track, dof_groups, TILTED_POSE, m_world)])
assert np.linalg.matrix_rank(J6, tol=1e-9) == 5   # exact rank 5/6 under pod_model='point'
U, _, _ = np.linalg.svd(J6)
# null direction is pure torque-about-moment-axis
assert abs(np.dot(U[:, -1], np.concatenate([np.zeros(3), m_hat]))) > 1 - 1e-6

# pod_model='multi' recovers yaw (rank 5/6 -> 6/6, tilted and dead level);
# the reconciled magnet-moment scale (/ N_points) still realises a real
# yaw target through CurrentAllocator.
mm_J6 = build_multi_dipole_wrench_jacobian(dof_track, dof_groups, dof_pod, TILT_R, TILTED_POSE)
assert np.linalg.matrix_rank(mm_J6, tol=1e-9) == 6   # pod_model='multi' should be full rank under tilt
R_level, pos_level = np.eye(3), np.array([0.0, 0.0, 0.040])
mm_J6_level = build_multi_dipole_wrench_jacobian(dof_track, dof_groups, dof_pod, R_level, pos_level)
assert np.linalg.matrix_rank(mm_J6_level, tol=1e-9) == 6   # full rank even at dead level

mm_track, _ = load_track_preset("Classic Track")
mm_cal = calibrate_levitation(mm_track, dof_pod, dof_pod_pts, SimConfig(), pod_xy=0.0)
assign_currents_for_pod(mm_track, pod_xy=0.0, pod_total_mass=dof_pod.TotalMass,
                        config=SimConfig(), current_scale=mm_cal.current_scale)
mm_current_limit = 5.0 * max(m.current for m in mm_track.Magnets)   # 5x headroom over the levitation current

# temporarily divide out the point-count so the per-magnet moment matches what
# build_multi_dipole_wrench_jacobian assumes. restored in the finally, always.
N_points = float(dof_pod_pts.shape[0])
orig_strengths = [m.magnetic_strength for m in dof_pod.Magnets]
for m in dof_pod.Magnets:
    m.magnetic_strength /= N_points
try:
    rec = CurrentAllocator(current_limit=mm_current_limit, fz_weight=100.0, include_torque=True,
                           torque_weight=1000.0, pod_model="multi")
    rec.allocate(Wrench(force=np.zeros(3), torque=np.array([0.0, 0.0, 0.01])),
                 mm_track, pos_level, pod=dof_pod, R=R_level)
finally:
    for m, s in zip(dof_pod.Magnets, orig_strengths):
        m.magnetic_strength = s
# realises >90% of a dead-level yaw target at the reconciled scale
assert rec.last_realized_torque[2] / 0.01 > 0.9

# group_by='row': every row's coils sit symmetrically about y=0, so one
# shared per-row current can't make net Fy there; per-coil grouping can.
row_groups = group_electromagnets_by_row(track)
assert len(row_groups) == 3 and all(len(g.magnet_indices) == 3 for g in row_groups)
J_row = build_effectiveness_jacobian(track, row_groups, POD_POS, M_VEC_TEST)
J_coil = build_effectiveness_jacobian(track, groups, POD_POS, M_VEC_TEST)
assert np.all(np.abs(J_row[1, :]) < 1e-9)   # group_by='row' Fy effectiveness must be exactly 0 at y=0
assert np.max(np.abs(J_coil[1, :])) > 1e-9   # group_by='magnet' breaks the row-level degeneracy

for m in track.Magnets:
    m.current = 0.05
F0 = realized_force(track, POD_POS, M_VEC_TEST)
CurrentAllocator(current_limit=1.0, fz_weight=100.0, group_by="row").allocate(
    Wrench(force=np.array([0.0, 0.05, 0.0])), track, POD_POS, M_VEC_TEST)
assert abs((realized_force(track, POD_POS, M_VEC_TEST) - F0)[1]) < 1e-9   # no Fy at y=0 with row grouping

print("All control tests passed.")
