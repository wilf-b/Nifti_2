# Verified Baseline: Ported Code Only

This folder contains **only the code ported 1-to-1 from baseline_code.py**. No extensions, no feedback control, no SO(3) attitude dynamics. Everything here is copied verbatim from Lewis's dissertation and proven to match the original via parity check.

## What's Included

**Physics kernels (verified identical to baseline):**
- `physics/field.py` — Elliptic-integral field calculations (@njit compiled, from Bella Mak's code)
- `physics/field_query.py` — Field evaluation at points
- `physics/ideal_field.py` — Analytic ideal fields for comparison
- `physics/dipole.py` — Dipole moment (inlined from baseline)

**Geometry and data:**
- `geometry/transforms.py` — Rotation matrices (Euler angles)
- `geometry/collection.py` — Magnet assembly and inertia
- `geometry/shapes.py` — Magnet classes
- `geometry/sampling.py` — Pod point-cloud generation

**I/O:**
- `data_io/presets.py` — Load track and pod geometry
- `data_io/sensor_data.py` — Load sensor validation data
- `data_io/field_data.py` — Save/load computed field grids
- `data_io/results.py` — Save current assignments

**Simulation:**
- `simulation/dynamics.py` — Rigid-body ODE (Euler angles, open-loop)
- `simulation/potential.py` — Potential energy grids (static, controlled, ideal)
- `simulation/sensor_check.py` — Validation against sensor data

**Plotting:**
- `plotting/geometry_plots.py` — 3D track and pod visualization
- `plotting/heatmaps.py` — Potential energy heatmaps
- `plotting/trajectory.py` — Trajectory visualization

**Control and UI:**
- `track/control.py` — Current assignment formula
- `main.py` — Menu interface (options 1–5 only, no closed-loop)
- `naming.py` — Run labeling
- `config.py` — Constants and paths

**Reference:**
- `baseline_code.py` — The original 1,861-line monolith (read-only)

## What's NOT Included

- `control/` — All feedback controllers (new, not ported)
- `geometry/quaternion.py` — SO(3) utilities (new)
- `simulation/calibration.py` — Levitation self-calibration (new, uncertain logic)
- `simulation/closed_loop*.py` — Closed-loop simulation (new, experimental)
- `tests/` — Test suite (new)
- `notebooks/` — Jupyter notebooks (new)

## Menu Options (Ported)

```python
python main.py
```

1. **Plot Track and Pod in 3D** — Geometry visualization
2. **Check simulation against sensor data** — Validation
3. **Plot potential energy** — Static, controlled, or ideal field
4. **Simulate pod movement (open-loop)** — Run dynamics on a precomputed grid
5. **Compare controlled vs ideal field** — Check against theory
6. **Quit**

## Parity Check

Verify that this code matches the original:

```bash
python notebooks/nblib/parity_check.py
```

This runs field kernels and open-loop dynamics head-to-head with `baseline_code.py` on thousands of test points. All outputs must be identical to machine precision.

## For Students

This is the clean baseline. Everything here is from the dissertation and has been proven correct. If you want to:

- **Extend the control system**: Read the dissertation, start fresh. Don't inherit the closed-loop experiment in the main submission—it's not confident in its recalibration logic.
- **Modify the physics**: The field kernels are from Bella Mak's elliptic-integral code and are well-tested. Change the geometry (presets in `data/Presets/`), then re-run the parity check.
- **Add a new simulation mode**: Build it on top of `simulation/dynamics.py::run_dynamics()`, which handles the ODE integration. See `simulation/sensor_check.py` for an example.

## Known Limitations

- **Attitude only in Euler angles**: SO(3) is not here; the baseline used Euler angles and only validates static attitude.
- **No feedback control**: The baseline had no controllers; to add any, build from scratch.
- **Dipole model**: Single-point dipole at the COM. The full package's multi-dipole model (not here) is more accurate but slower.

## File Structure

```
reduced/
├── baseline_code.py            (reference, untouched)
├── config.py
├── main.py
├── naming.py
├── geometry/
│   ├── transforms.py
│   ├── collection.py
│   ├── shapes.py
│   └── sampling.py
├── data_io/
│   ├── presets.py
│   ├── sensor_data.py
│   ├── field_data.py
│   └── results.py
├── physics/
│   ├── field.py
│   ├── field_query.py
│   ├── ideal_field.py
│   └── dipole.py
├── simulation/
│   ├── dynamics.py
│   ├── potential.py
│   └── sensor_check.py
├── plotting/
│   ├── geometry_plots.py
│   ├── heatmaps.py
│   └── trajectory.py
├── track/
│   └── control.py
├── data/                        (symlink to ../data)
└── README.md                    (this file)
```

---

**Contact the main package maintainers** if you find a discrepancy between this code and `baseline_code.py`. The parity check is the source of truth.
