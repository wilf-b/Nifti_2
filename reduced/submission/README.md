# Maglev Simulation

A Python simulation of a magnetic-levitation pod-and-track system.

I inherited `baseline_code.py` — a single 1,861-line script from Lewis's
dissertation work — and split it into this package with no logic changes.
That part is done and checked: every physics kernel in the tree is copied
from the original verbatim, and a parity check (below) proves the modular
version agrees with it to floating-point zero. Treat `baseline_code.py` as
the numeric reference if a result ever looks wrong.

On top of that I built a closed-loop controller: PD control, attitude
dynamics, self-calibrating levitation, the lot. That part is new, it is not
in the dissertation, and I want to be upfront about where it landed: most of
it does not work yet, in the sense of "this pod levitates and stays put." It
isn't broken code — the tests pass and the numbers are real — but the control
problem turned out to be harder than the time I had for it, and a couple of
the limits I ran into are structural rather than a tuning problem. See
**Status** below before you spend time re-deriving them.

I've tried to make the line between "verified baseline" and "my unfinished
control work" sharp enough that you can ignore the second one entirely if you
just want a trustworthy field/dynamics simulator to build on. See **Two
layers** and **If you just want the baseline** below.

## What it does

Models the magnetic interaction between a levitating pod (carrying permanent
magnets) and a track built from rows of electromagnetic coils: field
calculations, potential-energy landscapes, per-coil current allocation, and
pod trajectories both open-loop and closed-loop. The field maths uses
elliptic-integral kernels (Derby & Olbert formulation, originally via Bella
Mak's code), JIT-compiled with Numba because the pure-Python version is too
slow to run a sweep with.

## Two layers (and it's not just "old file vs new file")

I originally thought about this as "ported" vs "new," because that's what
`PROVENANCE.md` tracks line-by-line. But that's not actually the split that
matters if you're deciding what to trust. Some of the new code is solid —
independently validated, nothing hinges on it working out — and some of the
new code is the actual experiment that came up short. Those are different
things, and I had them conflated in an earlier draft of this README. The
split that matters is:

**Solid — ported or new, doesn't matter, you can build on it.** All the
ported baseline physics (field kernels, magnet geometry, track assembly,
potential-energy grids, plotting), plus three new pieces that turned out to
be correct and useful on their own merits: `geometry/quaternion.py` (the
SO(3) math, checked against conserved angular momentum through a tumble in
`notebooks/04_attitude_representation.ipynb`), `track/geometry.py`
(`TrackRow` — just a track-assembly convenience, no control claims attached),
and `physics/dipole.py` (the multi-dipole force/torque model — this is the
one piece of the control work I'd call an unambiguous win, see Status).

**The experiment — new, and mostly unsuccessful so far.** The feedback
control attempt specifically: the PD controllers (`control/xy_controller.py`,
`z_controller.py`, `attitude_controller.py`, `combined_controller.py`) and
the closed-loop drivers that run them (`simulation/closed_loop.py`,
`closed_loop_so3.py`). I built this because the original script had no
closed-loop control at all, and wanted to see whether the track as designed
could hold a pod in place under it. After three notebooks of poking at it,
the honest answer is "only partially, and not on the current hardware
layout, with gains that have never seen real rig data." Details in
**Status**.

Two things I initially lumped in with "the experiment" and had to pull back
out, because I checked the actual dependencies rather than going on feel:

- `control/allocation.py` (the current-allocation solver) and
  `control/base.py` (the `Wrench` type) aren't part of the feedback-loop
  attempt — they're the optimizer the PD loops sit on top of, and they're
  also exactly what the structural-limits analysis in notebook 02 runs on
  directly, multi-dipole result included.
- `simulation/calibration.py` isn't part of it either. I'd originally filed
  it under "the experiment" because it's new and sits next to the control
  code, but what it actually does is measure how far short the baseline
  current schedule falls of levitating — that measurement is itself one of
  the honest negative findings in **Status**, not an unproven patch. The
  multi-dipole demo in notebook 02 also calls it directly, to get a realistic
  current scale to test against.

| | Status | Files |
|---|---|---|
| Solid | ported | `geometry/transforms.py`, `shapes.py`, `collection.py`, `sampling.py` |
| Solid | ported | `physics/field.py`, `field_query.py`, `ideal_field.py` |
| Solid | ported | `track/control.py`, `simulation/potential.py`, `sensor_check.py` |
| Solid | ported | Euler-angle half of `simulation/dynamics.py`, all of `data_io/`, all of `plotting/` |
| Solid | ported, reference | `baseline_code.py` (root, untouched) |
| Solid | new, validated | `geometry/quaternion.py`, SO(3) half of `simulation/dynamics.py` |
| Solid | new, validated | `track/geometry.py`, `physics/dipole.py` |
| Solid | new, validated | `control/allocation.py`, `control/base.py`, `simulation/calibration.py` |
| The experiment | new, unsuccessful | `control/xy_controller.py`, `z_controller.py`, `attitude_controller.py`, `combined_controller.py` |
| The experiment | new, unsuccessful | `simulation/closed_loop.py`, `closed_loop_so3.py` |
| Shared infra | new | `config.py`, `naming.py`, `tests/_helpers.py` — needed everywhere, not physics themselves |

`main.py` straddles the line: menu options 1-5 are solid, options 6-7 call
into the control experiment. `PROVENANCE.md` still has the line-number-level
ported-vs-new detail if you need to diff something against the original
script — just don't read "new" there as "unproven." Most of the new code
isn't the experiment, and even inside `control/` the split isn't the whole
package, it's specifically the feedback loops.

## Status

Where the control layer actually stands, plainest version I can write:

- The baseline current schedule I inherited (`track/control.py`,
  "Eq. 18" in the dissertation writeup) **does not levitate the pod on its
  own**, on any preset I tested. The lift it produces is more than a
  thousand times smaller than the pod's weight. `calibrate_levitation`
  rescales it so the peak lift matches the weight, but that's a patch on top
  of a schedule that was never meant to self-levitate — it isn't a fix.
- With the height held artificially and the vertical loop engaged, the
  lateral PD loop does have local authority — it can correct a 6.4 mm offset.
  But the shipped gains (`kd = 5`) are under-damped against the baseline
  schedule's own anti-restoring stiffness (+42 N/m at centre) and the pod
  overshoots and diverges if you let the run go on. Nothing here has been
  tuned against real rig data — see `notebooks/03_baseline_control.ipynb`
  for the actual traces.
- With row-level actuation, the coil layout is exactly mirror-symmetric
  about y = 0, so lateral force and roll torque both vanish to
  floating-point zero at track centre. No current pattern gets around that;
  it's geometry, not gain-tuning.
- Point-dipole yaw torque is unreachable (rank 5/6) on every track/pose I
  tried: every coil axis is vertical, so `tau = m x B` is always
  perpendicular to `m`. The fix isn't more current, it's a different pod
  model — switching to the multi-point-dipole model already used for the
  real dynamics (`CurrentAllocator(pod_model="multi")`) recovers full 6/6
  rank through the same coils. That part I'd call a genuine result, not a
  dead end.

None of this is "the code is wrong" — the test suite passes and the parity
check against the original script is exact. It's "the actuator, as laid out
on the real track, can't do everything I wanted it to do, and the control
loop on top of it hasn't been tuned against anything physical." If you pick
this up, I'd start either from the multi-dipole result (it's the one piece
that actually extends what the hardware can do) or from re-deriving a
current schedule that levitates before worrying about the loops on top of
it.

See `notebooks/02_structural_limits.ipynb`, `notebooks/03_baseline_control.ipynb`,
`notebooks/04_attitude_representation.ipynb`, and `tests/test_control.py` for
the receipts behind every bullet above.

## If you just want the solid part, not the control experiment

If the closed-loop control attempt isn't useful to you, here's what to
actually drop. I checked this against the real imports rather than going off
`PROVENANCE.md`'s ported/new labels, because those don't line up with
solid/experimental — see **Two layers** above. This isn't packaged as a
second repo; the "drop" list below is short and self-contained enough that
deleting it in place (or copying everything else into a fresh directory)
gets you there without needing a second copy of the whole tree to maintain.

I actually did this in a scratch copy and ran the result before writing the
instructions below — my first two attempts at this list were wrong (I'd
mentally filed things as "experimental" because they were new and nearby,
not because they were actually part of the feedback-loop attempt), and the
naive version ("delete `control/` and `simulation/calibration.py`, trim
everything after the controller imports") broke the multi-dipole result and
left `main.py` unimportable. This is the version that actually runs.

**Drop these files:**
- `control/xy_controller.py`, `z_controller.py`, `attitude_controller.py`,
  `combined_controller.py` — **not** `control/allocation.py` or
  `control/base.py`, which stay
- `simulation/closed_loop.py`, `simulation/closed_loop_so3.py` — **not**
  `simulation/calibration.py`, which stays (see **Two layers** for why both
  of these turned out to be wrong calls on my first pass)
- `notebooks/03_baseline_control.ipynb` — the write-up of the PD-gain
  results specifically. Keep `notebooks/02_structural_limits.ipynb`: it runs
  on `control/allocation.py` and `simulation/calibration.py`, not the PD
  loops, and it's where the multi-dipole result lives.

**Edit `main.py`:** it imports all four dropped controller modules and both
closed-loop drivers unconditionally at the top, so it won't even start
without them, regardless of which menu option you pick. Remove the six
import lines (`simulation.closed_loop`, `closed_loop_so3`,
`control.xy_controller`, `attitude_controller`, `z_controller`,
`combined_controller`), remove `"Simulate pod movement (closed-loop x,y
control)"` and the full-stack one below it from the `actions` list, and
remove the `elif choice == 5:` and `elif choice == 6:` bodies (the one right
before `else: print("Goodbye.")` is the last one to go — nothing needs to
replace them, the surrounding `if/elif/else` chain is unaffected). I didn't
believe this would bite until I actually ran `python3 -c "import main"`
against the stripped tree and watched it die on the `closed_loop` import;
after the edit above it imports clean and the menu runs end-to-end with five
options plus Quit.

**Edit `tests/test_control.py`:** it interleaves allocation tests and
PD-controller tests throughout the file, it's not two clean halves. Remove
the four controller imports, then three self-contained blocks (each is
clearly marked by a comment, so this should survive line numbers drifting):
the lateral/vertical PD block (`lt = LateralTarget(...)` through the
`ZPDController` saturation check, right before the
`# attitude_controller:` comment), the attitude PD block (that comment
through the `sol_att` assert, right before `# torque Jacobian:`), and the
`CombinedController` composition block (`# CombinedController composition
...` through its assert, right before `# allocate() used to force...`).
Everything else in the file — the allocation/Jacobian checks, the torque
allocator, the Fz regression, the point-vs-multi-dipole rank test (the one
that calls `calibrate_levitation`), and the row-grouping test — stays, since
none of it is PD-controller code.

`tests/test_simulation.py` needs no changes: now that `calibration.py` is
staying, there's nothing in that file left to drop.

**Keep everything else**, including `geometry/quaternion.py`,
`track/geometry.py`, and `physics/dipole.py` — `tests/test_core.py`,
`test_physics.py`, and `_helpers.py` all import them directly for
non-experimental reasons (SO(3) math checks, track assembly, the dipole
force model), and they have nothing to do with whether the feedback loops
worked.

I ran `tests/test_core.py`, `test_physics.py`, the trimmed
`tests/test_control.py`, and `tests/test_simulation.py` against a stripped
copy of this tree with exactly the above removed, standalone, each with its
own Python process — all four pass. `python3 notebooks/nblib/parity_check.py`
also still applies unchanged against the stripped tree, since it only
touches the ported baseline functions.

## Layout

```
main.py              CLI entry point; the only place input() is called
config.py            physical constants (SI), output paths, SimConfig defaults
naming.py            RunTag dataclass, output-file naming (replaces 6 original globals)
baseline_code.py     original script, untouched, the numeric reference

geometry/
  transforms.py      rotation_matrix_from_euler + euler_from_matrix
  quaternion.py      SO(3) / quaternion utilities for attitude dynamics
  shapes.py          Magnet / PermMagnet / ElectroMagnet (pure geometry containers)
  collection.py      Collection (rings of magnets), inertia tensors
  sampling.py        generate_pod_base_points, volumetric dipole-point cloud
physics/
  field.py           elliptic-integral field kernels (@njit)
  field_query.py     field_at_points / field_from_collection / field_gradient_at_point
  dipole.py          dipole_moment, multi_dipole_force_torque, per_magnet_body_moments
  ideal_field.py     idealised valley / gravity field models
  ideal_field_diss.py    dissertation-exact ideal field (parity-only, unused by prod)
  linearise.py       finite-difference stiffness Jacobian (analysis only)
track/
  geometry.py        TrackRow dataclass, track assembly
  control.py         empirical baseline current schedule (assign_currents_for_pod)
control/
  base.py            Wrench (force / torque) + Controller base class
  xy_controller.py   lateral PD controller
  attitude_controller.py  geometric SO(3) roll / pitch PD controller
  z_controller.py    vertical PD controller
  combined_controller.py  sums Wrenches from a list of controllers
  allocation.py      desired Wrench -> per-group coil current corrections
data_io/             (named data_io, not io, to dodge the stdlib clash)
  presets.py         load_track_preset / load_pod_preset
  field_data.py      save / load field & energy grids (.xlsx)
  results.py         save currents, controlled-vs-ideal comparisons
  sensor_data.py     load_sensor_data (Train_mT2.csv)
simulation/
  dynamics.py        rigid-body dynamics (Euler-angle + SO(3)), open / closed-loop
  closed_loop.py     closed-loop driver (Euler-angle path, x, y control)
  closed_loop_so3.py closed-loop driver (quaternion / SO(3) path, full attitude)
  calibration.py     self-calibrating levitation current scale
  potential.py       energy_potential_static / _controlled / _ideal
  sensor_check.py    sensor validation against Train_mT2.csv
  dipole_oscillation.py, oscillation_analysis.py   dissertation Table 1/2 reproduction (unused by prod)
plotting/
  heatmaps.py        2D potential-energy heatmaps
  geometry_plots.py  3D track & pod cylinder plots
  trajectory.py      position-vs-time plots (Euler + SO(3))
notebooks/           01-04, one executed .ipynb per results section; nblib/ shared helpers
tests/               regression suite (test_core / test_physics / test_control / test_simulation)
data/                Presets/ and Sensor Data/, both checked in and required by the tests
outputs/             created on import; plots and saved grids land here
docs/results_figures/    created by the notebooks; their PNGs are written here
```

## Running it

```bash
python main.py               # interactive menu
python tests/run_all.py      # regression suite
```

### Menu (`main.py`)

| # | Option | Produces |
|---|---|---|
| 1 | Plot track and pod in 3D | 3D geometry plot |
| 2 | Check simulation against sensor data | field-vs-recorded comparison |
| 3 | Plot potential energy (Static / Controlled / Ideal) | heatmap + saved energy grid |
| 4 | Simulate pod movement | trajectory plot (builds a Controlled grid first if absent) |
| 5 | Compare controlled vs ideal field | comparison output |
| 6 | Simulate pod movement, closed-loop x, y | trajectory under lateral PD control |
| 7 | Simulate pod movement, closed-loop full (lateral + attitude + z) | SO(3) trajectory under the combined controller |
| 8 | Quit | — |

Roll / pitch / yaw are in radians. X / Y / Z ranges are full width in metres,
centred on 0 for X / Y and on the pod COM for Z. Output files are named by
`RunTag`, e.g. `[Controlled] Track (Classic Track) Track Current (Row 1: Default,
) Pod (Two Stacks [Theta 0_0, 0_0, 0_0])`, so each file records which preset,
angle and mode produced it.

## Verifying the build

1. **Regression suite.** `python3 tests/run_all.py`. Runs `test_core`,
   `test_physics`, `test_control`, `test_simulation` in dependency order (a
   subprocess each), stops at the first failure, and ends with
   `All test files passed.` Needs `data/Presets/*` and `data/Sensor Data/*`, both
   checked in. No pytest anywhere — each `tests/test_*.py` is a plain script of
   top-to-bottom asserts sharing `tests/_helpers.py`.

2. **Monolith parity.** `python3 notebooks/nblib/parity_check.py`. `check_kernels`
   runs the copied field / geometry leaf functions through both codebases on
   identical inputs (expect exact agreement, thousands of samples).
   `check_dynamics` runs `baseline_code.py`'s own `plot_movement` head-to-head
   with `simulation/dynamics.py::run_dynamics` on the same grid and state (all 12
   trajectory components agree to floating-point noise, through a developing
   instability). Prints a table, exits non-zero on mismatch. Deliberately **not**
   in `tests/run_all.py`, so `baseline_code.py` is never a test dependency.

## Tests

| File | Covers |
|---|---|
| `test_core.py` | `config`, `naming`, `geometry/`, `track/` |
| `test_physics.py` | `physics/` (field kernels, field_query, ideal_field, dipole), `data_io/` |
| `test_control.py` | `control/` (allocation, PD controllers) — the largest package |
| `test_simulation.py` | `simulation/` (SO(3) dynamics, calibration, open-loop grid path) |
| `_helpers.py` | shared fixtures and repo-root path bootstrap; not a test file |

Passing tests here mean the control layer's internal maths is self-consistent
— e.g. the allocator solves the current-allocation problem it's given
correctly — not that the controller levitates a real pod. Those are different
claims; see **Status**.

## Provenance

`PROVENANCE.md` is authoritative on what was ported from `baseline_code.py`
versus what is new, down to the line number: everything under `control/`,
the SO(3) attitude dynamics, the closed-loop simulation and levitation
self-calibration, `config.py`, `naming.py`, `track/geometry.py` and `tests/`
are new — the original had no feedback control, no attitude integration and
no tests. That's a different axis from solid-vs-experimental, though — see
**Two layers** for which parts of that "new" list you can actually trust.

## Reference

- Lewis's dissertation: the mathematical reference for the original script.
- Derby & Olbert / Bella Mak: the elliptic-integral field formulation.
- `data/Sensor Data/Train_mT2.csv`: sensor validation data showing the position of each of the coils.
