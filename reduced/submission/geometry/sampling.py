"""
Fill a Collection's magnet volume with a cloud of points, used to
discretise the pod into dipole points for the energy calculations.

Ported from baseline_code.py's generate_pod_base_points.
"""

import numpy as np

from geometry.collection import Collection


def generate_pod_base_points(pod, N=300):
    """
    Roughly N points sampled from inside the pod's magnets. Take the
    bounding box of all magnets, lay down a grid spaced so the box holds
    about N points scaled by (box volume / magnet volume), then keep only
    the grid points that land inside some magnet cylinder. Returns an
    (M, 3) array with M a bit under N.
    """
    x_mins, x_maxs = [], []
    y_mins, y_maxs = [], []
    z_mins, z_maxs = [], []

    for magnet in pod.Magnets:
        r  = magnet.radius
        h  = magnet.height
        cx, cy, cz = magnet.position

        x_mins.append(cx - r);  x_maxs.append(cx + r)
        y_mins.append(cy - r);  y_maxs.append(cy + r)
        z_mins.append(cz - h / 2); z_maxs.append(cz + h / 2)

    x_min, x_max = min(x_mins), max(x_maxs)
    y_min, y_max = min(y_mins), max(y_maxs)
    z_min, z_max = min(z_mins), max(z_maxs)

    Lx = x_max - x_min
    Ly = y_max - y_min
    Lz = z_max - z_min

    bbox_volume   = Lx * Ly * Lz
    density       = N / pod.TotalVolume
    target_points = density * bbox_volume

    # Grid spacing that gives roughly target_points in the bounding box
    d  = (bbox_volume / target_points) ** (1.0 / 3.0)
    nx = max(1, int(np.ceil(Lx / d)))
    ny = max(1, int(np.ceil(Ly / d)))
    nz = max(1, int(np.ceil(Lz / d)))

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)
    zs = np.linspace(z_min, z_max, nz)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    grid    = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

    # Keep points inside at least one magnet cylinder
    mask = np.zeros(len(grid), dtype=bool)
    for magnet in pod.Magnets:
        cx, cy, cz = magnet.position
        dx = grid[:, 0] - cx
        dy = grid[:, 1] - cy
        dz = grid[:, 2] - cz

        radial   = dx ** 2 + dy ** 2 <= magnet.radius ** 2
        vertical = np.abs(dz) <= magnet.height / 2
        mask    |= radial & vertical

    return grid[mask]
