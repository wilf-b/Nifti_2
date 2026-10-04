"""
Constants, paths and simulation defaults, collected in one place so
nothing else needs its own module globals. In baseline_code.py these
were scattered through the script.
"""

from pathlib import Path
from dataclasses import dataclass

import numpy as np


# physical constants (SI)

MU_0 = np.pi * 4e-7          # permeability of free space, H/m
MM = 1e-3                    # mm -> m
A_COIL = 9.5 * MM            # coil radius, m
A_WIRE = (0.25 / 2) * MM     # wire radius, m
DENSITY_COPPER = 8950.0      # kg/m^3
DENSITY_NDFEB = 7500.0       # kg/m^3, NdFeB permanent magnet
G = 9.81                     # m/s^2


# project paths

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data"
PRESETS_DIR = DATA_DIR / "Presets"
SENSOR_DATA_DIR = DATA_DIR / "Sensor Data"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_3D_TP_DIR = OUTPUT_DIR / "Plots" / "3D Plot Track & Pod"
PLOT_2D_HM_DIR = OUTPUT_DIR / "Plots" / "2D Plot Heatmap"
PLOT_2D_PT_DIR = OUTPUT_DIR / "Plots" / "2D Plot Position against Time"
PLOT_3D_PT_DIR = OUTPUT_DIR / "Plots" / "3D Plot Position against Time"
POTENTIAL_ENERGY_DIR = OUTPUT_DIR / "Saved_Magnetic_Field"
B_FIELD_DIR = OUTPUT_DIR / "Saved_Magnetic_Field" / "Save B Field"
IDEAL_COMPARISON_DIR = OUTPUT_DIR / "Ideal Comparison"
SENSOR_COMPARISON_DIR = OUTPUT_DIR / "Sensor Comparison"
SAVED_CURRENTS_DIR = OUTPUT_DIR / "savedCurrents"

# make the output folders on import so nothing downstream has to mkdir
for _folder in [PLOT_3D_TP_DIR, PLOT_2D_HM_DIR, PLOT_2D_PT_DIR, PLOT_3D_PT_DIR,
                POTENTIAL_ENERGY_DIR, B_FIELD_DIR, IDEAL_COMPARISON_DIR,
                SENSOR_COMPARISON_DIR, SAVED_CURRENTS_DIR]:
    _folder.mkdir(parents=True, exist_ok=True)


# simulation defaults

@dataclass
class SimConfig:
    # added by Wilf
    """Parameters for a potential-energy or dynamics run, passed to the
    simulation functions instead of prompting for each value."""

    # spatial grid
    x_min: float = -0.05
    x_max: float = 0.05
    x_num: int = 10

    y_min: float = -0.05
    y_max: float = 0.05
    y_num: int = 10

    z_min: float = 0.30
    z_max: float = 0.50
    z_num: int = 10

    # pod sampling
    pod_sample_points: int = 300   # N in generate_pod_base_points

    # dynamics (solve_ivp)
    t_span: tuple = (0.0, 2.0)
    max_step: float = 0.001

    # controlled-field calibration priors: empirical proportions that
    # were hardcoded in energy_potential_controlled / energy_potential_static.
    # Update these when the rig changes.
    reference_magnet_count: int = 78      # denominator for Magnet_proportion
    reference_pod_mass: float = 0.1459    # kg, denominator for Mass_proportion
    z_proportion: float = 1.0


# default instance used by all default runs; build a fresh SimConfig for a
# different experiment rather than mutating this one
DEFAULT_SIM_CONFIG = SimConfig()
