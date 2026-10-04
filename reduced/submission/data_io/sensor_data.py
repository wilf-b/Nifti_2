"""
Loader for the rig sensor data in Train_mT2.csv.

The CSV has 27 columns: x0,y0,z0 ... x7,y7,z7 are field readings at 8
sensor locations in mT, and podx,pody,podz are the pod position for each
row in metres. The sensor locations are hardcoded from the rig setup.

Ported from baseline_code.py's load_sensor_data. That version has a
pre-existing unit bug on the podx/pody/podz columns, fixed here, see the
load_sensor_data docstring.
"""

import numpy as np
import pandas as pd

from config import SENSOR_DATA_DIR


# physical sensor locations on the rig, in metres
SENSOR_LOCATIONS = np.array([
    [ 0.0125,  0.025,  0.012],
    [-0.0125,  0.025,  0.100],
    [ 0.0125,  0.000,  0.012],
    [-0.0123,  0.000,  0.012],
    [-0.0125,  0.075,  0.012],
    [ 0.0125,  0.075,  0.012],
    [-0.0125,  0.050,  0.012],
    [ 0.0125,  0.050,  0.012],
])


def load_sensor_data(filename="Train_mT2.csv"):
    """
    Read the sensor CSV in SENSOR_DATA_DIR and return

        sensor_locations : (8, 3)    fixed sensor positions, metres
        pod_positions    : (N, 3)    pod XYZ per measurement row, metres
        sensor_fields    : (N, 8, 3) measured B per sensor, Tesla

    The field columns (x0..z7) are mT and scaled to Tesla below.

    The podx/pody/podz columns are already in metres, not mm, even though
    this module's old docstring and baseline_code.py both applied a `*1e-3`.
    The raw values sit in the same range as SENSOR_LOCATIONS (podx around
    +/-0.006, pody 0.0125 to 0.075, podz 0.012 to 0.026); the `*1e-3`
    would shrink the whole sweep to a few tens of micrometres, below a
    wire diameter, which is not a real pod-to-sensor sweep. This is a bug
    in baseline_code.py itself (line 377), not introduced here, and it went
    unnoticed because this loader's file path was also broken, so the
    sensor check had never run end to end. Corrected here because this is
    a data loader, not a physics kernel.
    """
    path = SENSOR_DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Sensor data file not found: {path}")

    df = pd.read_csv(path)

    pod_positions = df[["podx", "pody", "podz"]].to_numpy()

    per_sensor_fields = []
    for i in range(8):
        per_sensor_fields.append(df[[f"x{i}", f"y{i}", f"z{i}"]].to_numpy())
    sensor_fields = np.stack(per_sensor_fields, axis=1) * 1e-3   # mT -> T

    return SENSOR_LOCATIONS, pod_positions, sensor_fields
