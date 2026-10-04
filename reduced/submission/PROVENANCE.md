# Provenance

What in this codebase came from `baseline_code.py` (Lewis's original 1,861-line baseline_code.py script) versus what's new. Line numbers refer to `baseline_code.py` as it stands.

## Ported

Same maths and logic as `baseline_code.py`, moved into the package and tidied. This was the fundamental aim of the original refactor - preserve all the features of the code verbatim and then come back to see what could be improved.


| Current location | From `baseline_code.py` | Notes |
|---|---|---|
| `geometry/transforms.py::rotation_matrix_from_euler` | `rotation_matrix_from_euler` (L62) | unchanged |
| `geometry/collection.py::Collection` | `class Collection` (L101-176) | `ChangeAngle`, `compute_inertia_tensor` ported as-is |
| `geometry/shapes.py::Magnet/PermMagnet/ElectroMagnet` | same classes (L177-285) | geometry/material fields only, `field_at()` removed |
| `data_io/presets.py::load_track_preset/load_pod_preset` | same (L286-371) | `input()` prompting stripped out |
| `data_io/sensor_data.py::load_sensor_data` | same (L372-383) | unchanged |
| `data_io/field_data.py` | `load/save_potential_energy`, `load/save_magnetic_field` (L384-669) | unchanged |
| `data_io/results.py::save_currents` | `savecurrents` (L670-700) | renamed |
| `physics/field_query.py` | `Full_Field_At_Point`, `Field_On_Pod_Point`, `rotate_point_into_pod_frame` (L701-724) + the `Magnet.field_at()` methods | renamed, pulled off the Magnet classes |
| `physics/ideal_field.py` | `B_field_Valley`, `B_field_Gravity` (L725-745) | unchanged |
| `physics/field.py` | `constants`, `B_calc`, `elliptic`, `B_rho`, `B_z`, `closed_cyl` (L746-892) | unchanged; originally from Bella Mak's code |
| `simulation/sensor_check.py` | `Sensor_Check`, `Ideal_Check` (L893-1036) | unchanged |
| `geometry/sampling.py::generate_pod_base_points` | same (L1037-1108) | unchanged |
| `simulation/potential.py` | `energy_potential_ideal/_controlled/_static` (L1109-1370) | shared loop extracted into `_compute_grid()` |
| `track/control.py::assign_currents_for_pod` | current-assignment formula inlined in `energy_potential_*` | pulled out to one place |
| `plotting/geometry_plots.py`, `plotting/heatmaps.py` | `plot_track_and_pod_3D`, `plot_heatmap_with_gradient` (L1371-1585) | unchanged |
| `simulation/dynamics.py` (Euler path) + `plotting/trajectory.py` | `plot_movement` (L1586-1751) | ODE integration split from plotting |
| `main.py` | `potential_choice`, `bad_input`, `int_choice`, `Interface` (L1752-1863) | menu logic unchanged, calls delegated elsewhere |

The regression suite tests the modular code's own internal consistency. The direct check against `baseline_code.py` is `scripts/parity/`: `monolith_parity_check.py` imports the original and diffs the field kernels and `rotation_matrix_from_euler` on identical inputs (exact 0.0, thousands of samples); `monolith_dynamics_parity_check.py` runs the original's `plot_movement` against  `simulation/dynamics.py::run_dynamics` on the same grid/state, and all twelve trajectory components agree through a developing instability. Both are standalone artifacts, deliberately not wired into `tests/run_all.py`.

## New

No equivalent in `baseline_code.py`.


The whole `control/` package, SO(3) attitude dynamics (`geometry/quaternion.py` and the SO(3) half of `simulation/dynamics.py`), closed-loop simulation and levitation self-calibration (`simulation/closed_loop*.py`,`simulation/calibration.py`, `physics/dipole.py`), `config.py`, `naming.py`,`track/geometry.py`, and `tests/`. The original had no feedback control, no attitude integration, and no tests.

`physics/ideal_field_diss.py`, `simulation/dipole_oscillation.py` and `simulation/oscillation_analysis.py` implement the dissertation's own
dipole-oscillation methodology fresh, however nothing in the production code imports/uses them.

## Dead / unused

- `track/control.py::assign_currents()`: the non-pod-aware variant, replaced
  by `assign_currents_for_pod()`. Zero callers, and its own docstring flags a
  likely-incorrect mass-proportion calculation.
