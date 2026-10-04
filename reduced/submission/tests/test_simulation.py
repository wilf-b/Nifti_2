"""
simulation/: SO(3) rigid-body dynamics, levitation calibration, and the
open-loop precomputed-grid path. Run with `python3 tests/test_simulation.py`.

The calibration and grid-interpolation sections need real preset files
under data/Presets/; without them those sections print a skip note and
carry on.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import synthetic_electro_track, angmom_and_ke, INERTIA_DIAG

import numpy as np
from scipy.integrate import solve_ivp

from geometry.collection import Collection
from geometry.shapes import PermMagnet
from geometry.sampling import generate_pod_base_points
from geometry.quaternion import quat_identity
from physics.field_query import field_gradient_at_point
from physics.dipole import dipole_moment
from track.control import assign_currents_for_pod
from config import G
from simulation.dynamics import _rigid_body_derivatives_so3

_I = INERTIA_DIAG
_I_inv = np.linalg.inv(_I)

# torque-free precession conserves world-frame L and KE; z is free-fall.
# dipole moment is zeros(3) here, so _B produces no torque and just rides
# along. the spin 2, 0.5, -1 is asymmetric so all three body axes are live.
_B = np.array([0.01, 0.02, -0.03])
_state0 = [0, 0, 0, 0, 0, 0, *quat_identity(), 2.0, 0.5, -1.0]
sol = solve_ivp(lambda t, s: _rigid_body_derivatives_so3(s, _B, np.zeros((3, 3)), np.zeros(3), 0.2, _I, _I_inv),
                (0, 5.0), _state0, max_step=0.001)   # 0.2 kg pod, 5 s, tight max_step to keep the integrator honest

L0, KE0 = angmom_and_ke(np.array(_state0), _I)
Lf, KEf = angmom_and_ke(sol.y[:, -1], _I)
assert abs(np.linalg.norm(sol.y[6:10, -1]) - 1.0) < 1e-6   # quaternion drifted from unit norm
assert np.linalg.norm(Lf - L0) / np.linalg.norm(L0) < 1e-6   # world-frame L not conserved
assert abs(KEf - KE0) / KE0 < 1e-6   # KE not conserved
assert np.isclose(sol.y[2, -1], -0.5 * G * 5.0 ** 2, rtol=1e-3)   # z should match free-fall

# angular_damping dissipates KE and |L| monotonically. _c small so the decay
# is visible over 5 s but doesn't stiffen the ODE; 11 sample points is enough
# to catch a non-monotonic step.
_c = 0.001
t_eval = np.linspace(0, 5.0, 11)
sol_d = solve_ivp(
    lambda t, s: _rigid_body_derivatives_so3(s, _B, np.zeros((3, 3)), np.zeros(3), 0.2, _I, _I_inv, angular_damping=_c),
    (0, 5.0), _state0, max_step=0.001, t_eval=t_eval)
_series = [angmom_and_ke(sol_d.y[:, i], _I) for i in range(sol_d.y.shape[1])]
KE_series = np.array([ke for _, ke in _series])
Lnorm_series = np.array([np.linalg.norm(L) for L, _ in _series])
assert np.all(np.diff(KE_series) < 1e-12)   # KE must decrease monotonically
assert np.all(np.diff(Lnorm_series) < 1e-12)   # |L| must decrease monotonically
assert KE_series[-1] < KE0
assert abs(np.linalg.norm(sol_d.y[6:10, -1]) - 1.0) < 1e-6   # quaternion drifted under damping

from simulation.calibration import calibrate_levitation, LevitationCalibration, local_current_scale

# error path: a deliberately feeble magnet (1e4, ~100x weaker than the presets)
# that can't lift anything, so calibration has to bail with a ValueError rather
# than return a bogus scale
small_track = synthetic_electro_track()
small_pod = Collection([PermMagnet(0.01, 0.005, np.array([0.0, 0.0, 0.0]), 1e4)])
small_pts = generate_pod_base_points(small_pod)
try:
    calibrate_levitation(small_track, small_pod, small_pts)
    assert False, "expected ValueError, fixture has no positive-lift height"
except ValueError as e:
    assert "positive lift" in str(e)
try:
    local_current_scale(small_track, np.array([0.0, 0.0, 0.03]), dipole_moment(small_pod, small_pts),
                        pod_total_mass=small_pod.TotalMass)
    assert False, "expected ValueError, no positive lift anywhere"
except ValueError as e:
    assert "positive lift" in str(e)

try:
    from data_io.presets import load_track_preset, load_pod_preset
    cal_track = load_track_preset("Classic Track")[0]
    cal_pod = load_pod_preset("Two Stacks")
    cal_pts = generate_pod_base_points(cal_pod)
    m_body = dipole_moment(cal_pod, cal_pts)
    weight = cal_pod.TotalMass * G

    cal = calibrate_levitation(cal_track, cal_pod, cal_pts)
    assert isinstance(cal, LevitationCalibration)
    assert cal.z_ref > 0 and cal.z_stable > cal.z_ref and cal.current_scale > 0

    assign_currents_for_pod(cal_track, pod_xy=0.0, pod_total_mass=cal_pod.TotalMass,
                            current_scale=cal.current_scale)

    def _Fz_at(track, pos):
        grad = np.zeros((3, 3))
        for m in track.Magnets:
            grad += field_gradient_at_point(m, pos)
        return float((m_body @ grad)[2])

    assert np.isclose(_Fz_at(cal_track, [0.0, 0.0, cal.z_ref]), cal.margin * weight, rtol=1e-4)
    assert np.isclose(_Fz_at(cal_track, [0.0, 0.0, cal.z_stable]), weight, rtol=1e-4)
    assert _Fz_at(cal_track, [0.0, 0.0, cal.z_stable + 1e-5]) - _Fz_at(cal_track, [0.0, 0.0, cal.z_stable]) < 0, \
        "dFz/dz at z_stable should be negative (restoring)"

    # local_current_scale reproduces cal.current_scale at the anchor,
    # differs away from it, and still makes Fz == weight there.
    anchor = local_current_scale(cal_track, np.array([0.0, 0.0, cal.z_stable]), m_body,
                                 cal_pod.TotalMass, margin=1.0)
    assert np.isclose(anchor, cal.current_scale, rtol=1e-3)
    # 2 cm off-axis, same height: far enough that the local scale genuinely
    # has to differ from the on-axis anchor
    offset_pos = np.array([0.0, 0.02, cal.z_stable])
    offset = local_current_scale(cal_track, offset_pos, m_body, cal_pod.TotalMass, margin=1.0)
    assert not np.isclose(offset, anchor, rtol=0.1)
    assign_currents_for_pod(cal_track, pod_xy=float(np.hypot(*offset_pos[:2])),
                            pod_total_mass=cal_pod.TotalMass, current_scale=offset)
    assert np.isclose(_Fz_at(cal_track, offset_pos), weight, rtol=1e-4)
except FileNotFoundError:
    print("   preset files not found, skipped the real-preset success-path check")

# run_dynamics grid path: a RegularGridInterpolator indexing bug would
# transpose gradient axes and fling the pod off. The grid is anisotropic
# (nx != ny != nz) so a transpose is detectable; a cube grid would hide it.
try:
    from data_io.presets import load_track_preset
    from simulation.dynamics import run_dynamics, DynamicsState
    from physics.field_query import field_from_collection

    gi_track = load_track_preset("Classic Track")[0]
    gi_pod = load_pod_preset("Two Stacks")
    gi_pts = generate_pod_base_points(gi_pod)
    assign_currents_for_pod(gi_track, pod_xy=0.0, pod_total_mass=gi_pod.TotalMass)

    # 9 x 7 x 5 - a different length on every axis, so a transposed lookup
    # can't accidentally line up and pass
    xs = np.linspace(-0.01, 0.01, 9)
    ys = np.linspace(-0.008, 0.008, 7)
    zs = np.linspace(0.030, 0.050, 5)
    Bx = np.empty((len(xs), len(ys), len(zs)))
    By = np.empty_like(Bx)
    Bz = np.empty_like(Bx)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            for k, z in enumerate(zs):
                Bx[i, j, k], By[i, j, k], Bz[i, j, k] = field_from_collection(gi_track, [[x, y, z]])[0]

    sol_gi = run_dynamics(gi_pod, gi_pts, [Bx, By, Bz], xs, ys, zs,
                          DynamicsState(x0=0.0, y0=0.0, z0=0.040),
                          t_span=(0.0, 0.02), max_step=0.002)
    end = sol_gi.y[:3, -1]
    assert np.all(np.abs(end[:2]) < 0.01) and abs(end[2] - 0.040) < 0.01, \
        f"pod flew off on the precomputed grid ({end}); likely a transposed grid axis"
except FileNotFoundError:
    print("   preset files not found, skipped")

print("All simulation tests passed.")
