#!/usr/bin/env python3
"""
Interactive CLI entry point. Every input() call lives here. Interface()
is the only thing worth importing; it drives the menu and hands the
actual work off to the rest of the package.
"""

import numpy as np

from data_io.presets import (
    list_track_presets, list_pod_presets,
    load_track_preset, load_pod_preset,
)
from data_io.field_data import load_magnetic_field
from geometry.collection import Collection
from geometry.sampling import generate_pod_base_points
from config import SimConfig, DEFAULT_SIM_CONFIG, B_FIELD_DIR
from naming import RunTag
from plotting.geometry_plots import plot_track_and_pod_3D
from plotting.heatmaps import plot_heatmap_with_gradient
from plotting.trajectory import plot_trajectory, plot_trajectory_so3
from simulation.dynamics import run_dynamics, DynamicsState, DynamicsStateSO3
from simulation.closed_loop import run_closed_loop_dynamics
from simulation.closed_loop_so3 import run_closed_loop_dynamics_so3
from simulation.calibration import calibrate_for_target, make_live_scale_fn
from simulation.potential import (
    energy_potential_static,
    energy_potential_controlled,
    energy_potential_ideal,
)
from simulation.sensor_check import sensor_check, ideal_check
from control.xy_controller import LateralTarget, LateralPDController
from control.attitude_controller import AttitudeTarget, GeometricAttitudePDController
from control.z_controller import ZTarget, ZPDController
from control.combined_controller import CombinedController
from control.allocation import CurrentAllocator


def _int_choice(prompt, choices):
    while True:
        print(prompt)
        for i, c in enumerate(choices):
            print(f"  {i+1} - {c}")
        try:
            idx = int(input("> ")) - 1
            if 0 <= idx < len(choices):
                return idx
            print("  Input out of range.")
        except ValueError:
            print("  Please enter an integer.")


def _ask_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("  Please enter a number.")


def _ask_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("  Please enter an integer.")


def _select_pod_and_track():
    # added by Wilf
    pod_names  = list_pod_presets()
    pod_name   = pod_names[_int_choice("Select pod:", pod_names)]
    pod        = load_pod_preset(pod_name)

    roll  = _ask_float("Roll  (rad): ")
    pitch = _ask_float("Pitch (rad): ")
    yaw   = _ask_float("Yaw   (rad): ")
    pod.ChangeAngle(roll, pitch, yaw)

    track_names           = list_track_presets()
    track_name            = track_names[_int_choice("Select track:", track_names)]
    track, currents_label = load_track_preset(track_name)

    tag = RunTag.from_euler(
        mode="[Controlled]", track_name=track_name,
        track_currents=currents_label, pod_name=pod_name,
        roll=roll, pitch=pitch, yaw=yaw,
    )
    pod_pts = generate_pod_base_points(pod)
    return pod, track, tag, pod_pts


def _ask_sim_config(pod, track):
    # added by Wilf
    x_range = _ask_float("X range (m, full width): ")
    x_num   = _ask_int("Points in X: ")
    y_range = _ask_float("Y range (m, full width): ")
    y_num   = _ask_int("Points in Y: ")

    z_track_top = max(m.position[2] + m.height / 2 for m in track.Magnets)
    print(f"\n  Pod COM z = {pod.Position[2]:.4f} m")
    print(f"  Track top = {z_track_top:.4f} m\n")

    z_range = _ask_float("Z range about Pod COM (m, full width): ")
    z_num   = _ask_int("Points in Z: ")

    return SimConfig(
        x_min=-x_range/2, x_max=x_range/2, x_num=x_num,
        y_min=-y_range/2, y_max=y_range/2, y_num=y_num,
        z_min=pod.Position[2] - z_range/2,
        z_max=pod.Position[2] + z_range/2,
        z_num=z_num,
    )


def Interface():
    # menu loop; the choice index lines up with `actions` below. Branches
    # 5 and 6 (closed-loop x,y and full lateral+attitude+z) are new on top
    # of baseline_code.py's original menu.
    actions = [
        "Plot Track and Pod in 3D",
        "Check simulation against sensor data",
        "Plot potential energy",
        "Simulate pod movement",
        "Compare controlled vs ideal field",
        "Simulate pod movement (closed-loop x,y control)",
        "Simulate pod movement (closed-loop full control: lateral + attitude + z)",
        "Quit",
    ]

    while True:
        choice = _int_choice("\n=== Main Menu ===", actions)

        if choice == 0:
            pod_names   = list_pod_presets()
            track_names = list_track_presets()
            pod_name    = pod_names[_int_choice("Select pod:", pod_names)]
            track_name  = track_names[_int_choice("Select track:", track_names)]
            pod          = load_pod_preset(pod_name)
            track, label = load_track_preset(track_name)
            roll  = _ask_float("Roll  (rad): ")
            pitch = _ask_float("Pitch (rad): ")
            yaw   = _ask_float("Yaw   (rad): ")
            pod.ChangeAngle(roll, pitch, yaw)
            tag = RunTag.from_euler("[Static]", track_name, label, pod_name, roll, pitch, yaw)
            plot_track_and_pod_3D(track, pod, tag)

        elif choice == 1:
            pod_names   = list_pod_presets()
            track_names = list_track_presets()
            pod_name    = pod_names[_int_choice("Select pod:", pod_names)]
            track_name  = track_names[_int_choice("Select track:", track_names)]
            pod          = load_pod_preset(pod_name)
            track, label = load_track_preset(track_name)
            tag = RunTag(mode="[Sensor]", track_name=track_name,
                         track_currents=label, pod_name=pod_name)
            sensor_check(pod, track, tag)

        elif choice == 2:
            pod, track, tag, pod_pts = _select_pod_and_track()
            cfg = _ask_sim_config(pod, track)
            mode = _int_choice("Select mode:", [
                "Static", "Controlled", "Ideal",
            ])
            modes = ["[Static]", "[Controlled]", "[Ideal]"]
            tag = RunTag(mode=modes[mode], track_name=tag.track_name,
                         track_currents=tag.track_currents,
                         pod_name=tag.pod_name, pod_angle=tag.pod_angle)
            if mode == 0:
                U = energy_potential_static(pod, track, pod.Position, pod_pts, tag, cfg)
            elif mode == 1:
                U = energy_potential_controlled(pod, track, pod.Position, pod_pts, tag, cfg)
            else:
                U = energy_potential_ideal(pod, pod.Position, pod_pts, tag, cfg)

            x_data = np.linspace(cfg.x_min, cfg.x_max, cfg.x_num)
            y_data = np.linspace(cfg.y_min, cfg.y_max, cfg.y_num)
            z_data = np.linspace(cfg.z_min, cfg.z_max, cfg.z_num)
            plot_heatmap_with_gradient(U, x_data, y_data, z_data, tag)

        elif choice == 3:
            pod, track, tag, pod_pts = _select_pod_and_track()
            cfg = _ask_sim_config(pod, track)
            ctrl_tag = RunTag(mode="[Controlled]", track_name=tag.track_name,
                              track_currents=tag.track_currents,
                              pod_name=tag.pod_name, pod_angle=tag.pod_angle)
            b_path = B_FIELD_DIR / f"{ctrl_tag.full_tag()}.xlsx"
            if not b_path.exists():
                print("No saved field, running the controlled sweep first...")
                energy_potential_controlled(pod, track, pod.Position, pod_pts, ctrl_tag, cfg)
            B_all, x_data, y_data, z_data = load_magnetic_field(b_path)
            print("\nStarting position:")
            state = DynamicsState(
                x0=_ask_float("x0 (m): "),
                y0=_ask_float("y0 (m): "),
                z0=_ask_float("z0 (m): "),
            )
            sol = run_dynamics(pod, pod_pts, B_all, x_data, y_data, z_data, state)
            plot_trajectory(sol, ctrl_tag)

        elif choice == 4:
            pod, track, tag, pod_pts = _select_pod_and_track()
            cfg = _ask_sim_config(pod, track)
            ctrl_tag  = RunTag(mode="[Controlled]", track_name=tag.track_name,
                               track_currents=tag.track_currents,
                               pod_name=tag.pod_name, pod_angle=tag.pod_angle)
            ideal_tag = RunTag(mode="[Ideal]", track_name=tag.track_name,
                               track_currents=tag.track_currents,
                               pod_name=tag.pod_name, pod_angle=tag.pod_angle)
            ideal_check(pod, track, pod.Position, pod_pts, ctrl_tag, ideal_tag, cfg)

        elif choice == 5:
            # closed-loop lateral (x, y) PD on a live field: no precomputed
            # grid, currents recomputed from the pod state each control step
            pod_names   = list_pod_presets()
            track_names = list_track_presets()
            pod_name    = pod_names[_int_choice("Select pod:", pod_names)]
            track_name  = track_names[_int_choice("Select track:", track_names)]
            pod          = load_pod_preset(pod_name)
            track, label = load_track_preset(track_name)
            pod_pts      = generate_pod_base_points(pod)

            tag = RunTag(mode="[ClosedLoop]", track_name=track_name,
                         track_currents=label, pod_name=pod_name)

            print("\nStarting position:")
            state = DynamicsState(
                x0=_ask_float("x0 (m): "),
                y0=_ask_float("y0 (m): "),
                z0=_ask_float("z0 (m): "),
            )
            print("\nTarget position (lateral only, z is not closed-loop in this phase):")
            target = LateralTarget(
                x=_ask_float("target x (m): "),
                y=_ask_float("target y (m): "),
            )

            # gains are untuned placeholders
            controller = LateralPDController(target=target, kp=20.0, kd=2.0,
                                             force_limit=0.15)
            allocator  = CurrentAllocator(current_limit=0.5, fz_weight=100.0)

            sol = run_closed_loop_dynamics(
                pod, pod_pts, track, controller, allocator, state,
                DEFAULT_SIM_CONFIG, t_span=(0.0, 1.0), control_dt=0.005,
            )
            plot_trajectory(sol, tag)

        elif choice == 6:
            # full closed-loop stack (lateral + SO(3) roll/pitch + z) over a
            # self-calibrating levitation baseline; yaw stays open-loop
            pod_names   = list_pod_presets()
            track_names = list_track_presets()
            pod_name    = pod_names[_int_choice("Select pod:", pod_names)]
            track_name  = track_names[_int_choice("Select track:", track_names)]
            pod          = load_pod_preset(pod_name)
            track, label = load_track_preset(track_name)
            pod_pts      = generate_pod_base_points(pod)

            print("\nStarting position:")
            x0 = _ask_float("x0 (m): ")
            y0 = _ask_float("y0 (m): ")
            print("\nStarting attitude (rad):")
            roll0  = _ask_float("roll0  (rad): ")
            pitch0 = _ask_float("pitch0 (rad): ")
            yaw0   = _ask_float("yaw0   (rad): ")

            print("\nTarget position:")
            target_x = _ask_float("target x (m): ")
            target_y = _ask_float("target y (m): ")
            print("\nTarget attitude (rad):")
            target_roll  = _ask_float("target roll  (rad): ")
            target_pitch = _ask_float("target pitch (rad): ")

            target_pod_xy = float(np.hypot(target_x, target_y))
            print(f"\nCalibrating baseline current scale at target pod_xy={target_pod_xy:.4f} ...")
            cal, current_limit = calibrate_for_target(track, pod, pod_pts, DEFAULT_SIM_CONFIG,
                                                       target_pod_xy)
            print(f"  z_stable={cal.z_stable:.5f} m  current_scale={cal.current_scale:.3f}"
                  f"  current_limit={current_limit:.4f} A")

            initial_state = DynamicsStateSO3.from_euler(
                x0=x0, y0=y0, z0=cal.z_stable, roll0=roll0, pitch0=pitch0, yaw0=yaw0,
            )

            tag = RunTag.from_euler(
                mode="[ClosedLoopFull]", track_name=track_name, track_currents=label,
                pod_name=pod_name, roll=roll0, pitch=pitch0, yaw=yaw0,
            )

            # gains here just show the combined stack runs; not tuned
            lateral_ctrl  = LateralPDController(target=LateralTarget(x=target_x, y=target_y),
                                                 kp=180.0, kd=5.0, force_limit=3.2)
            attitude_ctrl = GeometricAttitudePDController(
                target=AttitudeTarget(roll=target_roll, pitch=target_pitch),
                k_R=0.6, k_omega=0.0017, torque_limit=0.088,
            )
            z_ctrl = ZPDController(target=ZTarget(z=cal.z_stable),
                                    kp=5000.0, kd=20.0, force_limit=1.0)
            controller = CombinedController([lateral_ctrl, attitude_ctrl, z_ctrl])

            allocator = CurrentAllocator(current_limit=current_limit, fz_weight=100.0,
                                          include_torque=True, torque_weight=1000.0,
                                          tz_weight=100000.0, solver="bounded")
            scale_fn = make_live_scale_fn(track, pod, DEFAULT_SIM_CONFIG, allocator,
                                          cal.current_scale)

            # yaw is open-loop; the window is short because the uncalibrated
            # baseline's lateral force outgrows these gains as |pod_xy| rises
            print("\nRunning full closed-loop stack over t=(0.0, 0.1)s (control_dt=0.005s) ...")
            sol = run_closed_loop_dynamics_so3(
                pod, pod_pts, track, controller, allocator, initial_state, DEFAULT_SIM_CONFIG,
                t_span=(0.0, 0.1), control_dt=0.005, max_step=0.001,
                current_scale=scale_fn, angular_damping=0.01,
            )
            plot_trajectory_so3(sol, tag)

        else:
            print("Goodbye.")
            break


if __name__ == "__main__":
    Interface()
