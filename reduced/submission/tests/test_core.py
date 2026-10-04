"""
The dependency-free layers: config, naming.py, geometry/ and track/.
Run with `python3 tests/test_core.py`.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _helpers import synthetic_electro_track   # also puts the repo root on sys.path

import numpy as np

from config import MU_0, G, MM, DENSITY_NDFEB, SimConfig, OUTPUT_DIR
from naming import RunTag

# quick sanity that the constants survived the split from baseline_code.py intact
assert abs(MU_0 - np.pi * 4e-7) < 1e-20
assert abs(G - 9.81) < 1e-10 and abs(MM - 1e-3) < 1e-20
assert OUTPUT_DIR.exists()
assert SimConfig(x_num=5).x_num == 5   # kwargs actually land on the dataclass

tag = RunTag.from_euler("[Static]", "Track A", "Default", "Pod v1", 0.0, 0.0, 0.0).full_tag()
assert "Track A" in tag and "Pod v1" in tag and "[Static]" in tag

from geometry.transforms import rotation_matrix_from_euler
from geometry.shapes import PermMagnet, ElectroMagnet

assert np.allclose(rotation_matrix_from_euler(0, 0, 0), np.eye(3))
assert np.allclose(rotation_matrix_from_euler(0, 0, np.pi / 2) @ [1.0, 0, 0], [0, 1, 0], atol=1e-10)  # 90deg about z: +x -> +y

h, r = 0.01, 0.005   # 1 cm tall, 5 mm radius - the coil size most presets use
pm = PermMagnet(height=h, radius=r, position=[0, 0, 0], magnetic_strength=1e6)
assert abs(pm.volume - h * np.pi * r ** 2) < 1e-20
assert abs(pm.mass - DENSITY_NDFEB * pm.volume) < 1e-12
assert pm.magnetic_strength == 1e6 / MU_0   # gotcha: the ctor stores M/mu0, not the number you pass
assert pm.contains([[0.0, 0.0, 0.0]])[0] and not pm.contains([[0.1, 0.0, 0.0]])[0]

em = ElectroMagnet(height=0.01, radius=0.005, position=[0, 0, 0], current=1.0, turns=100)
assert em.turns == 100 and em.current == 1.0 and em.mass > 0

from geometry.collection import Collection
from geometry.sampling import generate_pod_base_points

# four identical magnets in a symmetric cross, so COM sits on the origin
c = Collection([
    PermMagnet(0.01, 0.005, [0.02, 0.0, 0.0], 1e6),
    PermMagnet(0.01, 0.005, [-0.02, 0.0, 0.0], 1e6),
    PermMagnet(0.01, 0.005, [0.0, 0.02, 0.0], 1e6),
    PermMagnet(0.01, 0.005, [0.0, -0.02, 0.0], 1e6),
])
assert abs(c.TotalMass - 4 * c.Magnets[0].mass) < 1e-12
assert abs(c.TotalVolume - 4 * c.Magnets[0].volume) < 1e-20
assert np.allclose(c.Position, [0, 0, 0], atol=1e-12)
I = c.compute_inertia_tensor()
assert np.allclose(I, I.T, atol=1e-12)   # inertia tensor is symmetric

# N is per-magnet and points inside overlaps get culled, so only bound loosely
pts = generate_pod_base_points(c, N=200)
assert pts.ndim == 2 and pts.shape[1] == 3 and 0 < len(pts) <= 200 * 3

from geometry.quaternion import (
    quat_identity, quat_to_matrix, matrix_to_quat, quat_multiply,
    quat_conjugate, quat_derivative, skew, vee,
)

assert np.allclose(quat_to_matrix(quat_identity()), np.eye(3))

# R -> quat -> R round-trip. generic pose first (nothing special about the angles,
# just not axis-aligned), then a pose a hair off gimbal lock where the naive
# euler path blows up but quats shouldn't.
R = rotation_matrix_from_euler(0.3, -0.2, 0.5)
assert np.max(np.abs(R - quat_to_matrix(matrix_to_quat(R)))) < 1e-9
R_gimbal = rotation_matrix_from_euler(0.01, np.pi / 2 - 1e-3, -0.02)
assert np.max(np.abs(R_gimbal - quat_to_matrix(matrix_to_quat(R_gimbal)))) < 1e-9

q1 = matrix_to_quat(rotation_matrix_from_euler(0.3, -0.2, 0.5))
q2 = matrix_to_quat(rotation_matrix_from_euler(-0.1, 0.4, 0.2))
R12 = rotation_matrix_from_euler(0.3, -0.2, 0.5) @ rotation_matrix_from_euler(-0.1, 0.4, 0.2)
assert np.allclose(R12, quat_to_matrix(quat_multiply(q1, q2)), atol=1e-9)
assert np.allclose(quat_multiply(q1, quat_conjugate(q1)), quat_identity(), atol=1e-9)

v = np.array([0.3, -1.2, 0.7])
assert np.allclose(vee(skew(v)), v)

# dt tiny on purpose: the first-order quat step should track exp(skew(w)*dt) to 1e-9
omega, dt = np.array([0.1, -0.2, 0.05]), 1e-6
q_next = quat_identity() + quat_derivative(quat_identity(), omega) * dt
assert np.allclose(quat_to_matrix(q_next), np.eye(3) + dt * skew(omega), atol=1e-9)

# body-frame inertia is rotation-invariant, world-frame isn't. three magnets
# strung along y, so Ixx/Iyy/Izz all differ and a bad transform shows up.
_pod = Collection([
    PermMagnet(0.01, 0.005, [0.0, -0.02, 0.0], 1e5),
    PermMagnet(0.01, 0.005, [0.0, 0.0, 0.0], 1e5),
    PermMagnet(0.01, 0.005, [0.0, 0.02, 0.0], 1e5),
])
I_world_0 = _pod.compute_inertia_tensor()
I_body = _pod.compute_body_inertia_tensor()
assert np.allclose(I_world_0, I_body)
_pod.ChangeAngle(0.3, -0.2, 0.5)
I_world_tilted = _pod.compute_inertia_tensor()
assert np.allclose(I_body, _pod.compute_body_inertia_tensor())
assert not np.allclose(I_world_0, I_world_tilted)
_R = rotation_matrix_from_euler(0.3, -0.2, 0.5)
assert np.allclose(I_body, _R.T @ I_world_tilted @ _R, atol=1e-9)

from track.geometry import TrackRow, track_row_to_magnets, track_rows_to_collection

perm_row = TrackRow(row_index=0, x_position=0.01, n_coils=3, spacing=0.02,
                    radius=0.005, height=0.01, coil_type="permanent", magnetic_strength=1e6)
perm_magnets = track_row_to_magnets(perm_row)
assert len(perm_magnets) == 3 and all(isinstance(m, PermMagnet) for m in perm_magnets)
# ring-building bug guard: y positions land on the symmetric grid (np.allclose, never ==)
assert np.allclose(sorted(m.position[1] for m in perm_magnets), [-0.02, 0.0, 0.02], atol=1e-10)

electro_row = TrackRow(1, -0.01, 2, 0.015, 0.005, 0.01, "electro", turns=100)   # positional args, exercises that path too
electro_magnets = track_row_to_magnets(electro_row)
assert len(electro_magnets) == 2 and all(isinstance(m, ElectroMagnet) for m in electro_magnets)
assert len(track_rows_to_collection([perm_row, electro_row]).Magnets) == 5

from track.control import assign_currents_for_pod

# start every coil at 0 A and let the empirical schedule fill them in.
# ref count/mass are set equal to the actual pod mass so the scaling cancels.
et = synthetic_electro_track(current=0.0)
assign_currents_for_pod(et, pod_xy=0.01, pod_total_mass=0.1459,
                        config=SimConfig(reference_magnet_count=4, reference_pod_mass=0.1459))
assert all(m.current > 0 and np.isfinite(m.current) for m in et.Magnets)

print("All core tests passed.")
