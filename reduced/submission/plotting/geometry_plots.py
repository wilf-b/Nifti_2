"""
3D visualisation of track and pod geometry.

Ported from baseline_code.py's plot_track_and_pod_3D.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from geometry.collection import Collection
from geometry.shapes import PermMagnet
from config import PLOT_3D_TP_DIR
from naming import RunTag


def _draw_cylinder(ax, centre, radius, height, angle,
                   color="gray", alpha=1.0, resolution=12):
    """Draw one cylinder on a 3D axis, oriented along `angle` (a unit
    vector) via Rodrigues' rotation formula."""
    x0, y0, z0 = centre

    direction = np.asarray(angle, dtype=float)
    direction /= np.linalg.norm(direction)

    theta     = np.linspace(0, 2 * np.pi, resolution)
    z_local   = np.linspace(0, height, resolution)
    Theta, Z  = np.meshgrid(theta, z_local)

    X = radius * np.cos(Theta)
    Y = radius * np.sin(Theta)

    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=0)

    z_axis = np.array([0.0, 0.0, 1.0])
    if not np.allclose(direction, z_axis):
        v  = np.cross(z_axis, direction)
        s  = np.linalg.norm(v)
        c  = np.dot(z_axis, direction)
        vx = np.array([
            [0,     -v[2],  v[1]],
            [v[2],   0,    -v[0]],
            [-v[1],  v[0],  0  ],
        ])
        R      = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2 + 1e-30))
        points = R @ points

    X = points[0].reshape(resolution, resolution) + x0
    Y = points[1].reshape(resolution, resolution) + y0
    Z = points[2].reshape(resolution, resolution) + z0

    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, edgecolor="none")


def plot_track_and_pod_3D(track, pod, tag, save=True, show=True):
    """
    3D plot of every track and pod magnet as a cylinder, PermMagnets in
    sky blue and ElectroMagnets in yellow. tag builds the filename; save
    writes the figure to PLOT_3D_TP_DIR; show calls plt.show().
    """
    fig = plt.figure(figsize=(10, 6))
    ax  = fig.add_subplot(111, projection="3d")

    for collection, label in [(track, "Track"), (pod, "Pod")]:
        for magnet in collection.Magnets:
            colour = "skyblue" if isinstance(magnet, PermMagnet) else "yellow"
            _draw_cylinder(
                ax,
                centre     = magnet.position,
                radius     = magnet.radius,
                height     = magnet.height,
                angle      = magnet.angle,
                color      = colour,
                alpha      = 0.8,
            )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"Pod COM: x={pod.Position[0]:.3f}, "
        f"y={pod.Position[1]:.3f}, z={pod.Position[2]:.3f}"
    )
    ax.set_box_aspect([1, 2, 0.5])
    ax.view_init(elev=20, azim=120)
    ax.grid(True)
    plt.tight_layout()

    if save:
        PLOT_3D_TP_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOT_3D_TP_DIR / f"{tag.full_tag()}.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
