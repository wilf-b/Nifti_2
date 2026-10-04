"""
Preset loading. From baseline_code.py's load_track_preset /
load_pod_preset with the input() prompting removed - the preset name
comes in as a string and main.py does the asking. Both .xlsx and .csv
files work, with a .csv alongside the .xlsx preferred (faster, no Excel
dependency).
"""

import numpy as np
import pandas as pd
from pathlib import Path

from config import MM
from config import PRESETS_DIR
from geometry.collection import Collection
from track.geometry import TrackRow, track_rows_to_collection
from geometry.shapes import PermMagnet


def _read_preset_file(filename_stem):
    # added by Wilf
    """Load a preset file by stem (e.g. 'TrackPresets', no extension),
    preferring CSV over xlsx when both exist."""
    csv_path  = PRESETS_DIR / f"{filename_stem}.csv"
    xlsx_path = PRESETS_DIR / f"{filename_stem}.xlsx"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    elif xlsx_path.exists():
        return pd.read_excel(xlsx_path)
    else:
        raise FileNotFoundError(
            f"No preset file found for '{filename_stem}' "
            f"(looked for {csv_path} and {xlsx_path})"
        )


def list_track_presets():
    # added by Wilf
    """Names of the available track presets."""
    df = _read_preset_file("TrackPresets")
    df = df.dropna(subset=["Setting Name"])
    return df["Setting Name"].tolist()


def load_track_preset(name):
    """
    Load a track preset by exact 'Setting Name' and return
    (collection, currents_label), where currents_label is the string for
    RunTag.track_currents. Raises ValueError if the name isn't found.
    """
    df      = _read_preset_file("TrackPresets")
    df      = df.dropna(subset=["Setting Name"])
    presets = df.to_dict(orient="records")

    matches = [p for p in presets if p["Setting Name"] == name]
    if not matches:
        raise ValueError(
            f"Track preset '{name}' not found. "
            f"Available: {[p['Setting Name'] for p in presets]}"
        )

    preset         = matches[0]
    n_rows         = int(preset["Rows"])
    currents_label = ""
    rows = []

    for i in range(1, n_rows + 1):
        coil_type_str = str(preset[f"Row {i} Type"]).lower()

        if "permanent" in coil_type_str:
            coil_type         = "permanent"
            magnetic_strength = float(preset[f"Row {i} Strength"])
            currents_label   += f"Row {i}: Default, "
            turns             = None
        elif "electro" in coil_type_str:
            coil_type         = "electro"
            magnetic_strength = None
            turns             = int(preset[f"Row {i} Turns"])
        else:
            raise ValueError(
                f"Unknown coil type '{coil_type_str}' in row {i} of preset '{name}'"
            )

        rows.append(TrackRow(
            row_index         = i - 1,
            x_position        = float(preset[f"Row {i} X"]) * MM,
            n_coils           = int(preset[f"Row {i} Coil Number"]),
            spacing           = float(preset[f"Row {i} Coil Seperation"]) * MM,
            radius            = float(preset[f"Row {i} Coil Radius"]) * MM,
            height            = float(preset[f"Row {i} Coil Height"]) * MM,
            coil_type         = coil_type,
            magnetic_strength = magnetic_strength,
            turns             = turns,
        ))

    return track_rows_to_collection(rows), currents_label


def list_pod_presets():
    # added by Wilf
    """Names of the available pod presets."""
    df = _read_preset_file("PodBasePresets")
    df = df.dropna(subset=["Setting Name"])
    return df["Setting Name"].tolist()


def load_pod_preset(name):
    """
    Load a pod preset by exact 'Setting Name' and return a Collection of
    PermMagnet objects. Raises ValueError if the name isn't found.
    """
    df      = _read_preset_file("PodBasePresets")
    df      = df.dropna(subset=["Setting Name"])
    matches = df[df["Setting Name"] == name]

    if matches.empty:
        raise ValueError(
            f"Pod preset '{name}' not found. "
            f"Available: {df['Setting Name'].tolist()}"
        )

    row     = matches.iloc[0]
    magnets = []

    for i in range(1, 13):
        x_val = row.get(f"X{i}", None)
        if x_val is None or pd.isna(x_val):
            continue

        magnets.append(
            PermMagnet(
                height            = float(row[f"Height {i}"]) * MM,
                radius            = float(row[f"Radius {i}"]) * MM,
                position          = np.array([
                    float(row[f"X{i}"]) * MM,
                    float(row[f"Y{i}"]) * MM,
                    float(row[f"Z{i}"]) * MM,
                ]),
                magnetic_strength = -float(row[f"Strength {i}"]),
            )
        )

    return Collection(magnets)
