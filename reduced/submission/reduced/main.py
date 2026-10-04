#!/usr/bin/env python3
"""
Interactive CLI entry point. Only the five ported menu options:
  1. Plot Track and Pod in 3D
  2. Check simulation against sensor data
  3. Plot potential energy
  4. Simulate pod movement (open-loop)
  5. Compare controlled vs ideal field
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
from plotting.trajectory import plot_trajectory
from simulation.dynamics import run_dynamics, DynamicsState
from simulation.potential import (
    energy_potential_static,
    energy_potential_controlled,
    energy_potential_ideal,
)
from simulation.sensor_check import sensor_check, ideal_check


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
    actions = [
        "Plot Track and Pod in 3D",
        "Check simulation against sensor data",
        "Plot potential energy",
        "Simulate pod movement (open-loop)",
        "Compare controlled vs ideal field",
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

        else:
            print("Goodbye.")
            break


if __name__ == "__main__":
    Interface()
