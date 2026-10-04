"""
Position and orientation vs time for a dynamics solution. The
integration (simulation/dynamics.py, simulation/closed_loop*.py) returns
a scipy OdeResult-compatible object that plot_trajectory takes. This is
the plotting half of baseline_code.py's plot_movement.
"""

import numpy as np
import matplotlib.pyplot as plt

from config import PLOT_2D_PT_DIR
from geometry.quaternion import quat_to_matrix, quat_normalize
from geometry.transforms import euler_from_matrix
from naming import RunTag


class _AsEulerView:
    # added by Wilf
    """Minimal .t/.y holder so plot_trajectory_so3 can hand a converted
    12-row array to plot_trajectory without it knowing about quaternions."""
    def __init__(self, t, y):
        self.t = t
        self.y = y
        self.t_events = []


def plot_trajectory(sol, tag, save=True, show=True):
    """
    Plot position (x, y, z) and orientation (roll, pitch, yaw) against
    time. sol is a scipy OdeResult (or compatible) from run_dynamics or
    run_closed_loop_dynamics, tag builds the filename, save writes to
    PLOT_2D_PT_DIR, show calls plt.show().
    """
    t     = sol.t
    x, y, z           = sol.y[0], sol.y[1], sol.y[2]
    roll, pitch, yaw  = sol.y[6], sol.y[7], sol.y[8]

    fig, axs = plt.subplots(3, 2, figsize=(10, 8), sharex=True)

    axs[0, 0].plot(t, x);     axs[0, 0].set_ylabel("x (m)")
    axs[1, 0].plot(t, y);     axs[1, 0].set_ylabel("y (m)")
    axs[2, 0].plot(t, z);     axs[2, 0].set_ylabel("z (m)")
    axs[2, 0].set_xlabel("Time (s)")

    axs[0, 1].plot(t, roll);  axs[0, 1].set_ylabel("roll (rad)")
    axs[1, 1].plot(t, pitch); axs[1, 1].set_ylabel("pitch (rad)")
    axs[2, 1].plot(t, yaw);   axs[2, 1].set_ylabel("yaw (rad)")
    axs[2, 1].set_xlabel("Time (s)")

    fig.suptitle(tag.full_tag())
    plt.tight_layout()

    if save:
        PLOT_2D_PT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PLOT_2D_PT_DIR / f"{tag.full_tag()}.png", dpi=150)

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_trajectory_so3(sol, tag, save=True, show=True):
    # added by Wilf
    """
    Plot an SO(3) closed-loop result by converting its quaternion
    columns (.y[6:10]) to Euler angles for display and reusing
    plot_trajectory. Display-only; the dynamics keeps the quaternion.
    """
    n     = sol.y.shape[1]
    euler = np.zeros((3, n))
    for i in range(n):
        q = quat_normalize(sol.y[6:10, i])
        euler[:, i] = euler_from_matrix(quat_to_matrix(q))

    y12 = np.vstack([sol.y[0:6], euler, sol.y[10:13]])
    plot_trajectory(_AsEulerView(sol.t, y12), tag, save=save, show=show)
