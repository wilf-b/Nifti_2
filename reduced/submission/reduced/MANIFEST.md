# Manifest: Files Ported 1-to-1 from baseline_code.py

This file lists the correspondence between `reduced/` files and the original `baseline_code.py` (1,861 lines).

## Physics Kernels (Verified by Parity Check)

| File | From baseline_code.py | Notes |
|------|---|---|
| `physics/field.py` | L746–892 (`B_calc`, `elliptic`, `B_rho`, `B_z`, `constants`, `closed_cyl`) | Unchanged, originally from Bella Mak |
| `physics/field_query.py` | L701–724 (`Full_Field_At_Point`, `Field_On_Pod_Point`, `rotate_point_into_pod_frame`) + Magnet methods | Renamed and refactored |
| `physics/ideal_field.py` | L725–745 (`B_field_Valley`, `B_field_Gravity`) | Unchanged |
| `physics/dipole.py` | L1132–1135, L1214, L1304, L1602 (inline dipole calculation) | Extracted to function |

## Geometry & Collections

| File | From baseline_code.py | Notes |
|------|---|---|
| `geometry/transforms.py` | L62 (`rotation_matrix_from_euler`) | Unchanged; `euler_from_matrix` added for display |
| `geometry/collection.py` | L101–176 (`class Collection`, `ChangeAngle`, `compute_inertia_tensor`) | `_build_body_inertia_tensor` added for SO(3) |
| `geometry/shapes.py` | L177–285 (`Magnet`, `PermMagnet`, `ElectroMagnet`) | Geometry/material only, `field_at()` removed |
| `geometry/sampling.py` | L1037–1108 (`generate_pod_base_points`) | Unchanged |

## I/O & Data

| File | From baseline_code.py | Notes |
|------|---|---|
| `data_io/presets.py` | L286–371 (`load_track_preset`, `load_pod_preset`) | `input()` prompting stripped out |
| `data_io/sensor_data.py` | L372–383 (`load_sensor_data`) | Unchanged |
| `data_io/field_data.py` | L384–669 (`load/save_potential_energy`, `load/save_magnetic_field`) | Unchanged |
| `data_io/results.py` | L670–700 (`savecurrents` → `save_currents`) | Renamed |

## Simulation & Analysis

| File | From baseline_code.py | Notes |
|------|---|---|
| `simulation/dynamics.py` | L1586–1751 (`plot_movement`) + L162–184 (Euler angle ODE) | ODE split from plotting; SO(3) path removed |
| `simulation/potential.py` | L1109–1370 (`energy_potential_ideal/_controlled/_static`) | Shared loop extracted into `_compute_grid()` |
| `simulation/sensor_check.py` | L893–1036 (`Sensor_Check`, `Ideal_Check`) | Unchanged |

## Visualization

| File | From baseline_code.py | Notes |
|------|---|---|
| `plotting/geometry_plots.py` | L1371–1585 (`plot_track_and_pod_3D`) | Subset only |
| `plotting/heatmaps.py` | L1371–1585 (`plot_heatmap_with_gradient`) | Subset only |
| `plotting/trajectory.py` | L1586–1751 (trajectory plotting from `plot_movement`) | Extracted for reuse |

## Control & UI

| File | From baseline_code.py | Notes |
|------|---|---|
| `track/control.py` | L25 (current assignment formula inlined) | Extracted to `assign_currents_for_pod()` |
| `main.py` | L1752–1863 (menu logic) + options 1–5 | Options 6–7 (closed-loop) removed; calls delegated |
| `config.py` | Constants scattered throughout, path setup | Collected for centralization |

## NOT Ported (New Code)

- `control/` — All feedback controllers, allocators, force computation
- `geometry/quaternion.py` — SO(3) and quaternion math
- `naming.py` — Run labeling (used by UI, minimal porting)
- `tests/` — Test suite
- `notebooks/` — Jupyter experiments

---

## Verification

Run the parity check to confirm this code is byte-identical to the original on thousands of test cases:

```bash
cd /path/to/parent/submission
python notebooks/nblib/parity_check.py
```

Expected output: all field kernels and dynamics components pass bit-for-bit comparison.
