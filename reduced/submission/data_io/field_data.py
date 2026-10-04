"""
Load and save computed field data: potential energy and magnetic field
grids.

Every save merges new data into any existing file for the same tag, so
runs accumulate incrementally, as the original did. The file format is
Excel with columns X_Pos, Y_Pos, Z_Pos, and either U_Val or Bx, By, Bz.

Ported from baseline_code.py's load_potential_energy, load_magnetic_field,
save_potential_energy(_comparison) and save_magnetic_field.
"""

import numpy as np
from pathlib import Path
import pandas as pd

from config import POTENTIAL_ENERGY_DIR, B_FIELD_DIR
from naming import RunTag


def load_potential_energy(path):
    """
    Read a saved potential energy grid from Excel. Returns
    (U_all, x_unique, y_unique, z_unique) with U_all shaped
    (Nx, Ny, Nz) and the coordinate arrays sorted.
    """
    data     = pd.read_excel(path)
    x_unique = np.sort(data["X_Pos"].unique())
    y_unique = np.sort(data["Y_Pos"].unique())
    z_unique = np.sort(data["Z_Pos"].unique())

    U_all = np.full((len(x_unique), len(y_unique), len(z_unique)), np.nan)
    x_idx = {x: i for i, x in enumerate(x_unique)}
    y_idx = {y: j for j, y in enumerate(y_unique)}
    z_idx = {z: k for k, z in enumerate(z_unique)}

    for _, row in data.iterrows():
        U_all[x_idx[row["X_Pos"]], y_idx[row["Y_Pos"]], z_idx[row["Z_Pos"]]] = row["U_Val"]

    return U_all, x_unique, y_unique, z_unique


def save_potential_energy(U_all, x_data, y_data, z_data, tag):
    """
    Save a potential energy grid to Excel, merging into any existing file
    with the same tag (new values win at matching coordinates). U_all is
    (Nx, Ny, Nz); tag is the RunTag used for the filename.
    """
    filename = tag.full_tag() + ".xlsx"
    path     = POTENTIAL_ENERGY_DIR / filename

    x_unique_new = np.sort(np.unique(x_data))
    y_unique_new = np.sort(np.unique(y_data))
    z_unique_new = np.sort(np.unique(z_data))

    # remap the incoming array onto unique-coordinate indexing
    U_new = np.full((len(x_unique_new), len(y_unique_new), len(z_unique_new)), np.nan)
    xi    = {x: i for i, x in enumerate(x_unique_new)}
    yi    = {y: j for j, y in enumerate(y_unique_new)}
    zi    = {z: k for k, z in enumerate(z_unique_new)}
    for i, x in enumerate(x_data):
        for j, y in enumerate(y_data):
            for k, z in enumerate(z_data):
                U_new[xi[x], yi[y], zi[z]] = U_all[i, j, k]

    if path.exists():
        U_old, x_old, y_old, z_old = load_potential_energy(path)
        x_comb = np.sort(np.unique(np.concatenate([x_old, x_unique_new])))
        y_comb = np.sort(np.unique(np.concatenate([y_old, y_unique_new])))
        z_comb = np.sort(np.unique(np.concatenate([z_old, z_unique_new])))
        U_comb = np.full((len(x_comb), len(y_comb), len(z_comb)), np.nan)

        xm = {x: i for i, x in enumerate(x_comb)}
        ym = {y: j for j, y in enumerate(y_comb)}
        zm = {z: k for k, z in enumerate(z_comb)}

        for i, x in enumerate(x_old):
            for j, y in enumerate(y_old):
                for k, z in enumerate(z_old):
                    if not np.isnan(U_old[i, j, k]):
                        U_comb[xm[x], ym[y], zm[z]] = U_old[i, j, k]

        for i, x in enumerate(x_unique_new):
            for j, y in enumerate(y_unique_new):
                for k, z in enumerate(z_unique_new):
                    if not np.isnan(U_new[i, j, k]):
                        U_comb[xm[x], ym[y], zm[z]] = U_new[i, j, k]
    else:
        x_comb, y_comb, z_comb, U_comb = x_unique_new, y_unique_new, z_unique_new, U_new

    X, Y, Z = np.meshgrid(x_comb, y_comb, z_comb, indexing="ij")
    df = pd.DataFrame({
        "X_Pos": X.ravel(), "Y_Pos": Y.ravel(),
        "Z_Pos": Z.ravel(), "U_Val": U_comb.ravel(),
    }).dropna(subset=["U_Val"])

    df.to_excel(path, index=False)


def load_magnetic_field(path):
    """
    Read a saved magnetic field grid from Excel. Returns
    (B_all, x_unique, y_unique, z_unique), where B_all is [Bx, By, Bz],
    each (Nx, Ny, Nz).
    """
    data     = pd.read_excel(path)
    x_unique = np.sort(data["X_Pos"].unique())
    y_unique = np.sort(data["Y_Pos"].unique())
    z_unique = np.sort(data["Z_Pos"].unique())
    shape    = (len(x_unique), len(y_unique), len(z_unique))

    B_all = [np.full(shape, np.nan) for _ in range(3)]
    x_idx = {x: i for i, x in enumerate(x_unique)}
    y_idx = {y: j for j, y in enumerate(y_unique)}
    z_idx = {z: k for k, z in enumerate(z_unique)}

    for _, row in data.iterrows():
        i, j, k = x_idx[row["X_Pos"]], y_idx[row["Y_Pos"]], z_idx[row["Z_Pos"]]
        B_all[0][i, j, k] = row["Bx"]
        B_all[1][i, j, k] = row["By"]
        B_all[2][i, j, k] = row["Bz"]

    return B_all, x_unique, y_unique, z_unique


def save_magnetic_field(B_all, x_data, y_data, z_data, tag):
    """
    Save a magnetic field grid to Excel, merging into any existing file
    with the same tag. B_all is [Bx, By, Bz], each (Nx, Ny, Nz); tag is
    the RunTag used for the filename.
    """
    filename = tag.full_tag() + ".xlsx"
    path     = B_FIELD_DIR / filename

    x_un = np.sort(np.unique(x_data))
    y_un = np.sort(np.unique(y_data))
    z_un = np.sort(np.unique(z_data))
    shape_new = (len(x_un), len(y_un), len(z_un))

    B_new = [np.full(shape_new, np.nan) for _ in range(3)]
    xi = {x: i for i, x in enumerate(x_un)}
    yi = {y: j for j, y in enumerate(y_un)}
    zi = {z: k for k, z in enumerate(z_un)}

    for i, x in enumerate(x_data):
        for j, y in enumerate(y_data):
            for k, z in enumerate(z_data):
                for c in range(3):
                    B_new[c][xi[x], yi[y], zi[z]] = B_all[c][i, j, k]

    if path.exists():
        B_old, x_old, y_old, z_old = load_magnetic_field(path)
        x_comb = np.sort(np.unique(np.concatenate([x_old, x_un])))
        y_comb = np.sort(np.unique(np.concatenate([y_old, y_un])))
        z_comb = np.sort(np.unique(np.concatenate([z_old, z_un])))
        shape_c = (len(x_comb), len(y_comb), len(z_comb))
        B_comb  = [np.full(shape_c, np.nan) for _ in range(3)]

        xm = {x: i for i, x in enumerate(x_comb)}
        ym = {y: j for j, y in enumerate(y_comb)}
        zm = {z: k for k, z in enumerate(z_comb)}

        for i, x in enumerate(x_old):
            for j, y in enumerate(y_old):
                for k, z in enumerate(z_old):
                    for c in range(3):
                        v = B_old[c][i, j, k]
                        if not np.isnan(v):
                            B_comb[c][xm[x], ym[y], zm[z]] = v

        for i, x in enumerate(x_un):
            for j, y in enumerate(y_un):
                for k, z in enumerate(z_un):
                    for c in range(3):
                        v = B_new[c][i, j, k]
                        if not np.isnan(v):
                            B_comb[c][xm[x], ym[y], zm[z]] = v
    else:
        x_comb, y_comb, z_comb = x_un, y_un, z_un
        B_comb = B_new

    X, Y, Z = np.meshgrid(x_comb, y_comb, z_comb, indexing="ij")
    df = pd.DataFrame({
        "X_Pos": X.ravel(), "Y_Pos": Y.ravel(), "Z_Pos": Z.ravel(),
        "Bx": B_comb[0].ravel(), "By": B_comb[1].ravel(), "Bz": B_comb[2].ravel(),
    }).dropna(subset=["Bx", "By", "Bz"], how="all")

    df.to_excel(path, index=False)
