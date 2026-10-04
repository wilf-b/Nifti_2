"""
Potential energy heatmaps with gradient arrows, on the three planes:
XY (fixed z), XZ (fixed y), YZ (fixed x).

Ported from baseline_code.py's plot_heatmap_with_gradient.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize

from config import PLOT_2D_HM_DIR
from naming import RunTag


def plot_heatmap_with_gradient(U_all, x_data, y_data, z_data, tag,
                               arrow_stride=2, arrow_scale=None,
                               save=True, show=True):
    """
    Potential energy heatmaps on the three orthogonal planes, with
    normalised gradient arrows on top. U_all is (Nx, Ny, Nz), tag builds
    the filename, arrow_stride plots one arrow every N grid points,
    arrow_scale is the quiver scale (None for auto), save writes to
    PLOT_2D_HM_DIR, show calls plt.show().
    """
    x_num, y_num, z_num = len(x_data), len(y_data), len(z_data)

    # colour map
    vmin, vmax = 0.0, 1.0
    value_colours = [
        (0.0, "darkblue"), (0.1, "lightblue"), (0.2, "yellow"),
        (0.4, "orange"),   (0.6, "red"),        (0.8, "purple"),
        (1.0, "black"),
    ]
    cmap = LinearSegmentedColormap.from_list(
        "fixed_map",
        [((v - vmin) / (vmax - vmin), c) for v, c in value_colours],
    )
    norm = Normalize(vmin=vmin, vmax=vmax)

    def _safe_extent(lo, hi):
        if lo == hi:
            eps = abs(lo) * 1e-6 + 1e-9
            return lo - eps, hi + eps
        return lo, hi

    x_lo, x_hi = _safe_extent(x_data.min(), x_data.max())
    y_lo, y_hi = _safe_extent(y_data.min(), y_data.max())
    z_lo, z_hi = _safe_extent(z_data.min(), z_data.max())

    # Gradient
    dU_dx = np.zeros_like(U_all)
    dU_dy = np.zeros_like(U_all)
    dU_dz = np.zeros_like(U_all)

    axes, spacing = [], []
    if x_num > 1: axes.append(0); spacing.append(x_data)
    if y_num > 1: axes.append(1); spacing.append(y_data)
    if z_num > 1: axes.append(2); spacing.append(z_data)

    if axes:
        grads   = np.gradient(U_all, *spacing, axis=axes, edge_order=2)
        g_index = 0
        if x_num > 1: dU_dx = grads[g_index]; g_index += 1
        if y_num > 1: dU_dy = grads[g_index]; g_index += 1
        if z_num > 1: dU_dz = grads[g_index]

    mag          = np.sqrt(dU_dx**2 + dU_dy**2 + dU_dz**2)
    mag[mag == 0] = 1.0
    dU_dx /= mag
    dU_dy /= mag
    dU_dz /= mag

    # Figure layout
    cols = max(x_num, y_num, z_num)
    fig  = plt.figure(figsize=(4 * cols, 10), constrained_layout=True)
    gs   = fig.add_gridspec(3, cols)

    s = arrow_stride

    # Row 0: XY planes at each fixed z
    for k in range(z_num):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(U_all[:, :, k].T, origin="lower",
                  extent=[x_lo, x_hi, y_lo, y_hi],
                  cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
        if x_num > 1 and y_num > 1:
            X, Y = np.meshgrid(x_data, y_data, indexing="ij")
            ax.quiver(X[::s, ::s], Y[::s, ::s],
                      -dU_dx[:, :, k][::s, ::s], -dU_dy[:, :, k][::s, ::s],
                      color="white", scale=arrow_scale)
        ax.set_title(f"z = {z_data[k]*1e3:.1f} mm")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    # Row 1: XZ planes at each fixed y
    for j in range(y_num):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(U_all[:, j, :].T, origin="lower",
                  extent=[x_lo, x_hi, z_lo, z_hi],
                  cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
        if x_num > 1 and z_num > 1:
            X, Z = np.meshgrid(x_data, z_data, indexing="ij")
            ax.quiver(X[::s, ::s], Z[::s, ::s],
                      -dU_dx[:, j, :][::s, ::s], -dU_dz[:, j, :][::s, ::s],
                      color="white", scale=arrow_scale)
        ax.set_title(f"y = {y_data[j]*1e3:.1f} mm")
        ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")

    # Row 2: YZ planes at each fixed x
    for i in range(x_num):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(U_all[i, :, :].T, origin="lower",
                  extent=[y_lo, y_hi, z_lo, z_hi],
                  cmap=cmap, norm=norm, interpolation="nearest", aspect="auto")
        if y_num > 1 and z_num > 1:
            Y, Z = np.meshgrid(y_data, z_data, indexing="ij")
            ax.quiver(Y[::s, ::s], Z[::s, ::s],
                      -dU_dy[i, :, :][::s, ::s], -dU_dz[i, :, :][::s, ::s],
                      color="white", scale=arrow_scale)
        ax.set_title(f"x = {x_data[i]*1e3:.1f} mm")
        ax.set_xlabel("Y (m)"); ax.set_ylabel("Z (m)")

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=fig.axes, shrink=0.95, label="Potential Energy (J)")
    fig.suptitle("Potential Energy with Gradient Field")

    if save:
        PLOT_2D_HM_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOT_2D_HM_DIR / f"{tag.full_tag()}.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)
