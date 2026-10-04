"""
physics/ (field kernels, field_query, ideal_field, dipole) and the
data_io/ sensor loader. Run with `python3 tests/test_physics.py`.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _helpers   # also puts the repo root on sys.path

import numpy as np

from config import MU_0
from geometry.shapes import PermMagnet, ElectroMagnet
from geometry.collection import Collection
from geometry.transforms import rotation_matrix_from_euler

from physics.field import B_calc, closed_cyl   # numba - the first call JITs, so it eats a second or two

# not checking values here, just that the kernel returns finite numbers for a
# plausible query. matching baseline_code's actual numbers is the parity check's job.
Bx, By, Bz = B_calc(x=0.005, y=0.0, z=0.02, I=0.0, a=0.005, L=0.01,
                    M=1e6 / MU_0, zcentre=0.0, b=0.005)
assert np.isfinite([Bx, By, Bz]).all()
pod_single = Collection([PermMagnet(0.01, 0.005, [0, 0, 0], 1e6)])
assert np.isfinite(closed_cyl(0.005, 0.0, 0.02, pod_single)).all()

from physics.field_query import field_at_points, field_from_collection

c = Collection([
    PermMagnet(0.01, 0.005, [0.02, 0.0, 0.0], 1e6),
    PermMagnet(0.01, 0.005, [-0.02, 0.0, 0.0], 1e6),
    PermMagnet(0.01, 0.005, [0.0, 0.02, 0.0], 1e6),
    PermMagnet(0.01, 0.005, [0.0, -0.02, 0.0], 1e6),
])
assert field_at_points(c.Magnets[0], [[0.0, 0.0, 0.05]]).shape == (1, 3)
assert field_from_collection(c, [[0.0, 0.0, 0.05]]).shape == (1, 3)
assert np.all(np.isfinite(field_from_collection(c, [[0.0, 0.0, 0.05]])))

from physics.ideal_field import B_field_valley, B_field_gravity

# 40 cm up is well clear of the track, so the ideal valley field is ~0 there
assert np.allclose(B_field_valley([0.0, 0.0, 0.40]), [0, 0, 0], atol=1e-10)
assert np.all(np.isfinite(B_field_gravity([0.0, 0.0, 0.40], total_mass=0.1, n_points=100)))

from physics.field_query import field_gradient_at_point

em = ElectroMagnet(height=0.01, radius=0.005, position=[0, 0, 0], current=2.0, turns=100)
off_axis = np.array([0.02, 0.01, 0.03])
grad = field_gradient_at_point(em, off_axis, h=1e-5)   # h is the central-difference step
assert grad.shape == (3, 3) and np.all(np.isfinite(grad))
# gradient should be roughly step-size independent across two orders of magnitude
assert np.allclose(field_gradient_at_point(em, off_axis, h=1e-6),
                   field_gradient_at_point(em, off_axis, h=1e-4), rtol=1e-2, atol=1e-2)
grad_axis = field_gradient_at_point(em, np.array([0.0, 0.0, 0.03]), h=1e-5)
assert abs(grad_axis[2, 0]) < 1e-6 and abs(grad_axis[2, 1]) < 1e-6   # dBz/dx,dy vanish on-axis

# multi_dipole collapses to sum(m_i x B) + sum((r_i-COM) x F_i) when the
# magnets share a point; the lever-arm term is zero in a uniform field.
from physics.dipole import multi_dipole_force_torque, per_magnet_body_moments

# arbitrary but fixed field + gradient so the analytic and the code path have
# something concrete to agree on
B0 = np.array([0.01, 0.02, 0.5])
GRAD = np.array([[0.3, -0.1, 0.05], [0.0, 0.2, -0.15], [0.1, 0.05, -0.5]])
R = rotation_matrix_from_euler(0.1, -0.2, 0.3)
COM = np.array([0.01, -0.02, 0.05])

# both magnets stacked at the same point => zero spread, so the lever-arm term
# has to collapse to a plain sum(m_i x B)
pos = np.array([0.005, 0.0, 0.02])
pod_deg = Collection([PermMagnet(0.01, 0.005, pos, 1.0), PermMagnet(0.01, 0.005, pos, 0.6)])
F_multi, tau_multi = multi_dipole_force_torque(
    pod_deg, R, COM, lambda p: B0 + GRAD @ p, lambda p: GRAD)
m_world = R @ sum(per_magnet_body_moments(pod_deg))
r_world = COM + R @ (pos - pod_deg.Position)
assert np.allclose(F_multi, m_world @ GRAD, rtol=1e-10, atol=1e-14)
assert np.allclose(tau_multi, np.cross(m_world, B0 + GRAD @ r_world), rtol=1e-10, atol=1e-14)

# now a real +/-3 cm spread in y; in a uniform field the lever arms still cancel
pod_spread = Collection([
    PermMagnet(0.01, 0.005, np.array([0.0, -0.03, 0.02]), 1.0),
    PermMagnet(0.01, 0.005, np.array([0.0, 0.03, 0.02]), 1.0),
])
F_u, tau_u = multi_dipole_force_torque(
    pod_spread, R, COM, lambda p: B0, lambda p: np.zeros((3, 3)))
assert np.allclose(F_u, 0.0, atol=1e-14)
assert np.allclose(tau_u, np.cross(R @ sum(per_magnet_body_moments(pod_spread)), B0),
                   rtol=1e-10, atol=1e-14)   # lever-arm term vanishes in a uniform field

# data_io.sensor_data: podx/pody/podz unit-bug guard. The old *1e-3 mm
# assumption made recorded pod positions 1000x too small; they are real
# rig coordinates in metres, same order as the fixed sensor locations.
try:
    from data_io.sensor_data import load_sensor_data, SENSOR_LOCATIONS

    sensor_locations, pod_positions, sensor_fields = load_sensor_data()
    assert sensor_locations.shape == (8, 3)
    assert pod_positions.shape[1] == 3 and pod_positions.shape[0] > 0
    assert sensor_fields.shape == (pod_positions.shape[0], 8, 3)

    sensor_scale = np.abs(SENSOR_LOCATIONS).max()
    pod_scale = np.abs(pod_positions).max()
    assert pod_scale > sensor_scale / 100, (
        f"pod_positions scale ({pod_scale:.2e} m) implausibly small next to "
        f"SENSOR_LOCATIONS ({sensor_scale:.2e} m); podx/pody/podz unit bug regressed"
    )
except FileNotFoundError:
    print("   sensor data file not found, skipped")

print("All physics tests passed.")
