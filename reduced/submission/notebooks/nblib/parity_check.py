"""
Checks the modular package still reproduces baseline_code.py.

check_kernels  - the copied field/geometry kernels, called through both code
                 paths on the same inputs.
check_dynamics - one full trajectory, run through baseline_code.plot_movement
                 and simulation.dynamics.run_dynamics.

Run: python3 notebooks/nblib/parity_check.py  (prints a table, exits 1 on
mismatch). Kept out of tests/run_all.py so baseline_code.py is never a
dependency of the committed suite.
"""

import sys
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOL = 1e-9


def load_baseline():
    """Import baseline_code.py without running its __main__ block."""
    spec = importlib.util.spec_from_file_location("baseline_code", ROOT / "baseline_code.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_kernels(bc):
    """{name: max |baseline - modular|} over a sweep of realistic inputs."""
    from physics.field import constants, elliptic, B_rho, B_z, closed_cyl
    from geometry.transforms import rotation_matrix_from_euler
    from data_io.presets import load_pod_preset

    rng = np.random.default_rng(0)
    n = 2000
    pod = load_pod_preset("Two Stacks")  # closed_cyl needs PermMagnets

    def constants_args():
        rho = rng.uniform(0.0, 0.05, n); rho[0] = 0.0  # rho = 0 edge case
        z = rng.uniform(-0.1, 0.1, n)
        a = rng.uniform(0.005, 0.02, n)
        M = rng.uniform(1e4, 1e6, n)
        b = rng.uniform(1e-3, 1e-2, n)
        return zip(rho, z, a, M, b)

    def elliptic_args():
        kc = np.concatenate([rng.uniform(1e-6, 1.0 - 1e-10, n),
                             [1e-6, 1.0 - 1e-10, 5e-7, 1.0 - 1e-11]])  # both clamp thresholds
        s = rng.choice([-1.0, 1.0], size=len(kc))
        return ((k, 1.0, 1.0, si) for k, si in zip(kc, s))

    def B_rho_args():
        B0 = rng.uniform(1e-3, 1.0, n)
        ap, am = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
        kp = rng.uniform(1e-6, 1.0 - 1e-10, n)
        km = rng.uniform(1e-6, 1.0 - 1e-10, n)
        return zip(B0, ap, kp, am, km)

    def B_z_args():
        B0 = rng.uniform(1e-3, 1.0, n)
        a = rng.uniform(0.005, 0.02, n)
        rho = rng.uniform(0.0, 0.05, n)
        bp, bm = rng.uniform(-1, 1, n), rng.uniform(-1, 1, n)
        kp = rng.uniform(1e-6, 1.0 - 1e-10, n)
        km = rng.uniform(1e-6, 1.0 - 1e-10, n)
        g = rng.uniform(-1, 1, n)
        return zip(B0, a, rho, bp, kp, g ** 2, g, bm, km)

    def closed_cyl_args():
        xs = rng.uniform(-0.05, 0.05, 200)
        ys = rng.uniform(-0.05, 0.05, 200)
        zs = rng.uniform(0.0, 0.06, 200)
        return ((x, y, z, pod) for x, y, z in zip(xs, ys, zs))

    def rot_args():
        roll = rng.uniform(-np.pi, np.pi, 500)
        pitch = rng.uniform(-np.pi / 2, np.pi / 2, 500)
        yaw = rng.uniform(-np.pi, np.pi, 500)
        return zip(roll, pitch, yaw)

    specs = [
        ("constants", bc.constants, constants, constants_args),
        ("elliptic", bc.elliptic, elliptic, elliptic_args),
        ("B_rho", bc.B_rho, B_rho, B_rho_args),
        ("B_z", bc.B_z, B_z, B_z_args),
        ("closed_cyl", bc.closed_cyl, closed_cyl, closed_cyl_args),
        ("rotation_matrix_from_euler", bc.rotation_matrix_from_euler, rotation_matrix_from_euler, rot_args),
    ]

    worst = {}
    for name, f_old, f_new, args in specs:
        d = 0.0
        for a in args():
            diff = np.abs(np.asarray(f_old(*a), float) - np.asarray(f_new(*a), float))
            d = max(d, float(diff.max()))
        worst[name] = d
    return worst


def check_dynamics(bc):
    """{state component: max |baseline - modular|} for one trajectory, t <= 0.1 s."""
    import shutil
    import matplotlib
    from unittest.mock import patch
    from scipy.integrate import solve_ivp as real_solve_ivp

    from config import SimConfig, B_FIELD_DIR
    from naming import RunTag
    from data_io.presets import load_track_preset, load_pod_preset
    from data_io.field_data import load_magnetic_field
    from geometry.sampling import generate_pod_base_points
    from simulation.potential import energy_potential_controlled
    from simulation.dynamics import run_dynamics, DynamicsState

    TRACK, POD = "Hex Track", "Two Stacks Conic Decreasing"
    X0, Y0, Z0 = 0.0, 0.0, 0.04536          # dissertation Table 3 Test 1
    T_COMPARE, MAX_STEP = 0.1, 0.001        # window ends before the ~0.13 s instability
    cfg = SimConfig(x_min=-0.05, x_max=0.05, x_num=5,
                    y_min=-0.05, y_max=0.05, y_num=5,
                    z_min=0.025, z_max=0.065, z_num=6)

    track, currents_label = load_track_preset(TRACK)
    pod = load_pod_preset(POD)
    pod.ChangeAngle(0.0, 0.0, 0.0)
    pod_pts = generate_pod_base_points(pod)

    tag = RunTag.from_euler(mode="[Controlled]", track_name=TRACK,
                            track_currents=currents_label, pod_name=POD,
                            roll=0.0, pitch=0.0, yaw=0.0)
    grid_path = B_FIELD_DIR / f"{tag.full_tag()}.xlsx"
    if grid_path.exists():
        grid_path.unlink()  # save_magnetic_field merges into an existing file; start clean
    energy_potential_controlled(pod, track, pod.Position, pod_pts, tag, cfg)
    B_all, x_data, y_data, z_data = load_magnetic_field(grid_path)

    # plot_movement looks its grid up by a shorter tag; hand it the same bytes.
    bc.Track_Used, bc.Pod_Used = TRACK, POD
    bc.Pod_Angle = "[Theta 0_0, 0_0, 0_0]"
    bc_tag = f"[Controlled] Track ({TRACK}) Pod ({POD} {bc.Pod_Angle})"
    shutil.copy(grid_path, bc.B_Field_Data_Folder / f"{bc_tag}.xlsx")

    # grid bounds from the loaded arrays, not cfg (a stale file could differ).
    gx = (float(x_data[0]), float(x_data[-1]), len(x_data))
    gy = (float(y_data[0]), float(y_data[-1]), len(y_data))
    gz = (float(z_data[0]), float(z_data[-1]), len(z_data))

    sol_mod = run_dynamics(pod, pod_pts, B_all, x_data, y_data, z_data,
                           DynamicsState(x0=X0, y0=Y0, z0=Z0),
                           t_span=(0.0, 0.3), max_step=MAX_STEP)

    # plot_movement returns nothing and draws/saves a figure; capture its
    # solve_ivp result and stub the interactive/plotting side effects.
    captured = {}

    def capture(fun, t_span, y0, **kw):
        captured["sol"] = real_solve_ivp(fun, t_span, y0, **kw)
        return captured["sol"]

    backend = matplotlib.get_backend()
    matplotlib.use("Agg", force=True)
    try:
        with patch.object(bc, "solve_ivp", capture), \
             patch.object(bc, "save_current_figure", lambda *a, **k: None), \
             patch("builtins.input", side_effect=[str(X0), str(Y0), str(Z0)]):
            bc.plot_movement(pod, track, pod.Position, pod_pts, *gx, *gy, *gz)
    finally:
        matplotlib.use(backend, force=True)
    sol_bc = captured["sol"]

    labels = ["x (m)", "y (m)", "z (m)", "roll (rad)", "pitch (rad)", "yaw (rad)"]
    rows = [0, 1, 2, 6, 7, 8]

    def window(sol):
        m = sol.t <= T_COMPARE
        return sol.t[m], sol.y[:, m]

    t_mod, y_mod = window(sol_mod)
    t_bc, y_bc = window(sol_bc)
    return {lab: float(np.max(np.abs(y_bc[i] - np.interp(t_bc, t_mod, y_mod[i]))))
            for lab, i in zip(labels, rows)}


def main():
    bc = load_baseline()
    results = {**check_kernels(bc), **check_dynamics(bc)}

    width = max(len(k) for k in results)
    print(f"\n{'check':<{width}}   max |baseline - modular|")
    for name, d in results.items():
        print(f"{name:<{width}}   {d:.2e}")

    worst = max(results.values())
    if worst > TOL:
        print(f"\nFAIL: {worst:.2e} exceeds {TOL:g}. Investigate before citing parity.")
        return 1
    print(f"\nPASS: everything agrees with baseline_code.py to within {TOL:g}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
