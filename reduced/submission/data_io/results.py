"""
Save simulation results that aren't field grids: magnet currents at each
pod position, and the ideal-vs-controlled potential energy comparison.

Ported from baseline_code.py's savecurrents and
save_potential_energy_comparison.
"""

import numpy as np
import pandas as pd

from config import SAVED_CURRENTS_DIR, IDEAL_COMPARISON_DIR
from geometry.collection import Collection
from naming import RunTag


def save_currents(track, pod_x, pod_y, pod_z, magnet_currents_xyz):
    """
    Write per-pod-position magnet currents to Excel. track gives the
    magnet positions; pod_x/pod_y/pod_z are the pod position at each grid
    point; magnet_currents_xyz is one current list per grid point.
    """
    SAVED_CURRENTS_DIR.mkdir(parents=True, exist_ok=True)
    file_path = SAVED_CURRENTS_DIR / "magnet_currents.xlsx"

    magnet_ids = list(range(len(track.Magnets)))
    magnet_x   = [m.position[0] for m in track.Magnets]
    magnet_y   = [m.position[1] for m in track.Magnets]

    magnet_df   = pd.DataFrame({
        "Magnet_ID":  magnet_ids,
        "x_position": magnet_x,
        "y_position": magnet_y,
    })

    currents_df = pd.DataFrame(magnet_currents_xyz)
    currents_df.insert(0, "Pod_z", pod_z)
    currents_df.insert(0, "Pod_y", pod_y)
    currents_df.insert(0, "Pod_x", pod_x)
    currents_df.columns = (
        ["Pod_x", "Pod_y", "Pod_z"]
        + [f"Magnet_{i}_Current" for i in range(len(track.Magnets))]
    )

    with pd.ExcelWriter(file_path) as writer:
        magnet_df.to_excel(writer,   sheet_name="Magnet_Positions", index=False)
        currents_df.to_excel(writer, sheet_name="Magnet_Currents",  index=False)


def save_potential_energy_comparison(x_data, y_data, z_data,
                                     U_controlled, U_ideal, U_comparison, tag):
    """
    Save a side-by-side controlled-vs-ideal potential energy comparison.
    U_comparison is U_ideal - U_controlled (NaN where either is NaN); tag
    is the RunTag used to build the filename.
    """
    IDEAL_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    file_path = IDEAL_COMPARISON_DIR / f"{tag.comparison_tag()}.xlsx"

    X, Y, Z = np.meshgrid(x_data, y_data, z_data, indexing="ij")

    df = pd.DataFrame({
        "x":            X.flatten(),
        "y":            Y.flatten(),
        "z":            Z.flatten(),
        "U_val":        U_controlled.flatten(),
        "U_val_ideal":  U_ideal.flatten(),
        "U_comparison": U_comparison.flatten(),
    })

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Grid_Data", index=False)

    print(f"Saved comparison to {file_path}")
