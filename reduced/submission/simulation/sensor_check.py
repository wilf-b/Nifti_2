"""
Validation against real sensor data.

sensor_check compares the simulated magnetic field against the recorded
field at 8 sensor locations. ideal_check runs both the controlled and
ideal sweeps and saves the comparison.

Ported from baseline_code.py's Sensor_Check and Ideal_Check.
"""

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

from geometry.collection import Collection
from physics.field_query import track_field_at_point
from physics.field import closed_cyl
from data_io.sensor_data import load_sensor_data
from data_io.field_data import load_potential_energy
from data_io.results import save_potential_energy_comparison
from simulation.potential import energy_potential_controlled, energy_potential_ideal
from config import SENSOR_COMPARISON_DIR, POTENTIAL_ENERGY_DIR
from config import SimConfig, DEFAULT_SIM_CONFIG
from naming import RunTag


def sensor_check(pod, track, tag):
    """
    Compare the simulated field against the recorded sensor data. For
    each pod position in the dataset, place the pod there, compute the
    field at all 8 sensor locations, and difference against the recording.
    tag builds the output filename. Returns a DataFrame with x, y, z
    columns holding the mean field difference (simulated minus recorded)
    per pod position.
    """
    sensor_locations, pod_positions, sensor_fields = load_sensor_data()
    z_base = max(magnet.height for magnet in track.Magnets)

    recorded_vectors  = []
    simulated_vectors = []

    iterator = (
        tqdm(enumerate(pod_positions), total=len(pod_positions),
             desc="Sensor check")
        if _HAS_TQDM
        else enumerate(pod_positions)
    )

    for i, pod_position in iterator:
        # shift the pod to the measurement position
        for magnet in pod.Magnets:
            magnet.position += pod_position
            magnet.position[2] += z_base

        recorded_block  = []
        simulated_block = []

        for j, (x, y, z) in enumerate(sensor_locations):
            B_sim = (
                np.array(track_field_at_point(track, (x, y, z)))
                + np.array(closed_cyl(x, y, z, pod))
            )
            B_rec = sensor_fields[i][j]
            simulated_block.append(B_sim)
            recorded_block.append(B_rec)

        simulated_vectors.append(simulated_block)
        recorded_vectors.append(recorded_block)

        # put the pod back at the origin
        for magnet in pod.Magnets:
            magnet.position -= pod_position
            magnet.position[2] -= z_base

    def _mean_df(blocks):
        rows = []
        for block in blocks:
            arr = np.array(block)          # (8, 3)
            rows.append({"x": arr[:, 0].mean(),
                         "y": arr[:, 1].mean(),
                         "z": arr[:, 2].mean()})
        return pd.DataFrame(rows)

    df_recorded  = _mean_df(recorded_vectors)
    df_simulated = _mean_df(simulated_vectors)
    df_diff      = df_simulated - df_recorded

    summary = {
        "Recorded_Mean":  df_recorded.mean(),
        "Recorded_Std":   df_recorded.std(),
        "Simulated_Mean": df_simulated.mean(),
        "Simulated_Std":  df_simulated.std(),
    }

    out_path = SENSOR_COMPARISON_DIR / f"{tag.full_tag()}_sensor_comparison.xlsx"
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df_recorded.to_excel(writer,  sheet_name="Recorded",   index=False)
        df_simulated.to_excel(writer, sheet_name="Simulated",  index=False)
        df_diff.to_excel(writer,      sheet_name="Difference", index=False)
        pd.DataFrame(summary).to_excel(writer, sheet_name="Stats")

    print(f"Saved sensor comparison to {out_path}")
    return df_diff


def ideal_check(pod, track, pod_pos, pod_pts, controlled_tag, ideal_tag,
                cfg=DEFAULT_SIM_CONFIG):
    """
    Run the controlled and ideal sweeps, then save a side-by-side
    comparison. pod_pos is the (3,) reference pod position, pod_pts the
    (N, 3) sample points; controlled_tag and ideal_tag are RunTags with
    mode "[Controlled]" and "[Ideal]".
    """
    # controlled sweep, then load the result
    energy_potential_controlled(pod, track, pod_pos, pod_pts,
                                controlled_tag, cfg)
    ctrl_path = POTENTIAL_ENERGY_DIR / f"{controlled_tag.full_tag()}.xlsx"
    U_ctrl, x_ctrl, y_ctrl, z_ctrl = load_potential_energy(ctrl_path)

    # ideal sweep, then load the result
    energy_potential_ideal(pod, pod_pos, pod_pts, ideal_tag, cfg)
    ideal_path = POTENTIAL_ENERGY_DIR / f"{ideal_tag.full_tag()}.xlsx"
    U_ideal, x_ideal, y_ideal, z_ideal = load_potential_energy(ideal_path)

    # matching coordinates between the two grids
    def _matching_indices(a, b):
        ia, ib = [], []
        for i, v in enumerate(a):
            matches = np.where(np.isclose(b, v, rtol=1e-12))[0]
            if len(matches):
                ia.append(i); ib.append(matches[0])
        return ia, ib

    ix_c, ix_i = _matching_indices(x_ctrl, x_ideal)
    iy_c, iy_i = _matching_indices(y_ctrl, y_ideal)
    iz_c, iz_i = _matching_indices(z_ctrl, z_ideal)

    shape = (len(ix_c), len(iy_c), len(iz_c))
    U_new      = np.zeros(shape)
    U_new_ideal= np.zeros(shape)
    U_comp     = np.zeros(shape)

    x_new = x_ctrl[ix_c]
    y_new = y_ctrl[iy_c]
    z_new = z_ctrl[iz_c]

    for i1, (i2, i3) in enumerate(zip(ix_c, ix_i)):
        for j1, (j2, j3) in enumerate(zip(iy_c, iy_i)):
            for k1, (k2, k3) in enumerate(zip(iz_c, iz_i)):
                a = U_ctrl[i2, j2, k2]
                b = U_ideal[i3, j3, k3]
                U_new[i1, j1, k1]       = a
                U_new_ideal[i1, j1, k1] = b
                U_comp[i1, j1, k1]      = (b - a) if (np.isfinite(a) and np.isfinite(b)) else np.nan

    comparison_tag = RunTag(
        mode           = "[Comparison]",
        track_name     = controlled_tag.track_name,
        track_currents = controlled_tag.track_currents,
        pod_name       = controlled_tag.pod_name,
        pod_angle      = controlled_tag.pod_angle,
    )

    save_potential_energy_comparison(
        x_new, y_new, z_new,
        U_new, U_new_ideal, U_comp,
        comparison_tag,
    )
