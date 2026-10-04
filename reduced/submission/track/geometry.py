"""
Track geometry, separate from the IO that loads it. data_io/presets.py
reads the Excel and builds TrackRow objects; this turns those into
Magnet instances and Collections. The track is rows of coils along X
with Y offsets from the row spacing. This layer wraps track-assembly
code that used to sit inside baseline_code.py's load_track_preset.
"""

import numpy as np
from dataclasses import dataclass

from geometry.shapes import PermMagnet, ElectroMagnet, Magnet
from geometry.collection import Collection
from config import MM


@dataclass
class TrackRow:
    # added by Wilf
    """
    One row of coils, at the same X and evenly spaced along Y. row_index
    is 0-based; x_position/spacing/radius/height in metres; coil_type is
    'permanent' or 'electro'; magnetic_strength (A/m) for permanent,
    turns for electro.
    """
    row_index:        int
    x_position:       float
    n_coils:          int
    spacing:          float
    radius:           float
    height:           float
    coil_type:        str
    magnetic_strength: float | None = None
    turns:            int   | None  = None


def track_row_to_magnets(row):
    # added by Wilf
    """
    A TrackRow to a list of PermMagnet / ElectroMagnet. Y positions are
    evenly-spaced offsets around zero, the original spacing convention.
    """
    y_offsets = (np.arange(row.n_coils) - (row.n_coils - 1) / 2) * row.spacing

    magnets = []
    for y in y_offsets:
        position = np.array([row.x_position, y, 0.0])

        if row.coil_type == 'permanent':
            magnet = PermMagnet(
                height=row.height,
                radius=row.radius,
                position=position,
                magnetic_strength=row.magnetic_strength,
            )
        elif row.coil_type == 'electro':
            magnet = ElectroMagnet(
                height=row.height,
                radius=row.radius,
                position=position,
                current=0.0,
                turns=row.turns,
            )
        else:
            raise ValueError(
                f"Unknown coil_type '{row.coil_type}' in row {row.row_index}. "
                "Expected 'permanent' or 'electro'."
            )

        # the allocator groups actuators per row, so tag the source row
        magnet.row_id = row.row_index
        magnets.append(magnet)

    return magnets


def track_rows_to_collection(rows):
    # added by Wilf
    """Build one Collection holding every magnet from a list of TrackRows."""
    all_magnets = []
    for row in rows:
        all_magnets.extend(track_row_to_magnets(row))
    return Collection(all_magnets)
