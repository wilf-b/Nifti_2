"""
Potential energy sweeps, three modes on one inner loop:
energy_potential_static holds the currents fixed (set at pod_xy=0),
energy_potential_controlled updates them per pod XY, energy_potential_ideal
uses the analytic field and needs no track. Each returns U_all and
saves/plots via its tag. From baseline_code.py's
energy_potential_static/_controlled/_ideal, with the shared loop factored
out.
"""

import numpy as np

from geometry.collection import Collection
from geometry.transforms import rotation_matrix_from_euler
from physics.field_query import track_field_at_point
from physics.ideal_field import B_field_valley, B_field_gravity
from physics.field import closed_cyl
from track.control import assign_currents_for_pod
from data_io.field_data import save_potential_energy, save_magnetic_field
from data_io.results import save_currents
from config import SimConfig, DEFAULT_SIM_CONFIG
from naming import RunTag
from physics.dipole import dipole_moment

try:
    from tqdm import tqdm
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


def _progress(iterable, total, desc):
    # added by Wilf
    """Wrap an iterable in tqdm when it's installed, else pass it through."""
    if _HAS_TQDM:
        return tqdm(iterable, total=total, desc=desc, unit="pt")
    return iterable


def _make_grid(cfg):
    # added by Wilf
    """Coordinate arrays from a SimConfig, cell-centre convention (linspace
    from min+dx/2 to max-dx/2)."""
    def centres(lo, hi, n):
        d = (hi - lo) / n
        return np.linspace(lo + d / 2, hi - d / 2, n)

    return (
        centres(cfg.x_min, cfg.x_max, cfg.x_num),
        centres(cfg.y_min, cfg.y_max, cfg.y_num),
        centres(cfg.z_min, cfg.z_max, cfg.z_num),
    )


def _any_inside(track, pts):
    # added by Wilf
    """True if any point in pts lies inside any track magnet."""
    for magnet in track.Magnets:
        if np.any(magnet.contains(pts)):
            return True
    return False


def energy_potential_static(pod, track, pod_pos, pod_pts, tag,
                            cfg=DEFAULT_SIM_CONFIG):
    """
    Potential energy sweep with fixed currents, set once at pod_xy = 0.
    pod_pos is the (3,) reference pod position (for the grid shift),
    pod_pts the (N, 3) pod-frame sample points, tag the RunTag for
    filenames. Returns a (x_num, y_num, z_num) array.
    """
    x_data, y_data, z_data = _make_grid(cfg)
    nx, ny, nz = cfg.x_num, cfg.y_num, cfg.z_num

    U_all = np.zeros((nx, ny, nz))
    B_all = [np.zeros((nx, ny, nz)) for _ in range(3)]
    m_vec = dipole_moment(pod, pod_pts)

    # currents set once at pod_xy = 0
    assign_currents_for_pod(track, pod_xy=0.0,
                            pod_total_mass=pod.TotalMass, config=cfg)

    total = nx * ny * nz
    bar   = _progress(range(nz), total=total, desc="Static potential")

    for k, z in enumerate(z_data):
        grav_term = pod.TotalMass * 9.81 * z
        for i, x in enumerate(x_data):
            for j, y in enumerate(y_data):
                shift      = np.array([x - pod_pos[0],
                                       y - pod_pos[1],
                                       z - pod_pos[2]])
                pts_shifted = pod_pts + shift

                if _any_inside(track, pts_shifted):
                    U_all[i, j, k] = np.nan
                else:
                    U_all[i, j, k] = grav_term
                    for pt in pts_shifted:
                        Bx, By, Bz = track_field_at_point(track, pt)
                        B_all[0][i, j, k] += Bx
                        B_all[1][i, j, k] += By
                        B_all[2][i, j, k] += Bz
                        U_all[i, j, k]    -= (m_vec[0]*Bx
                                              + m_vec[1]*By
                                              + m_vec[2]*Bz)
        if _HAS_TQDM:
            bar.update(ny * nx)

    if _HAS_TQDM:
        bar.close()

    save_potential_energy(U_all, x_data, y_data, z_data, tag)
    save_magnetic_field(B_all, x_data, y_data, z_data, tag)

    return U_all


def energy_potential_controlled(pod, track, pod_pos, pod_pts, tag,
                                cfg=DEFAULT_SIM_CONFIG):
    """
    Potential energy sweep with the currents updated per pod XY position.
    Arguments and return as energy_potential_static.
    """
    x_data, y_data, z_data = _make_grid(cfg)
    nx, ny, nz = cfg.x_num, cfg.y_num, cfg.z_num

    U_all = np.zeros((nx, ny, nz))
    B_all = [np.zeros((nx, ny, nz)) for _ in range(3)]
    m_vec = dipole_moment(pod, pod_pts)

    pod_x_list = []
    pod_y_list = []
    pod_z_list = []
    currents_list = []

    total = nx * ny * nz
    bar   = _progress(range(nz), total=total, desc="Controlled potential")

    for k, z in enumerate(z_data):
        grav_term = pod.TotalMass * 9.81 * z
        for i, x in enumerate(x_data):
            for j, y in enumerate(y_data):
                pod_xy = np.hypot(x, y)
                assign_currents_for_pod(track, pod_xy=pod_xy,
                                        pod_total_mass=pod.TotalMass, config=cfg)

                pod_x_list.append(x)
                pod_y_list.append(y)
                pod_z_list.append(z)
                currents_list.append([m.current for m in track.Magnets])

                shift       = np.array([x - pod_pos[0],
                                        y - pod_pos[1],
                                        z - pod_pos[2]])
                pts_shifted = pod_pts + shift

                if _any_inside(track, pts_shifted):
                    U_all[i, j, k] = np.nan
                else:
                    U_all[i, j, k] = grav_term
                    for pt in pts_shifted:
                        Bx, By, Bz = track_field_at_point(track, pt)
                        B_all[0][i, j, k] += Bx
                        B_all[1][i, j, k] += By
                        B_all[2][i, j, k] += Bz
                        U_all[i, j, k]    -= (m_vec[0]*Bx
                                              + m_vec[1]*By
                                              + m_vec[2]*Bz)
        if _HAS_TQDM:
            bar.update(ny * nx)

    if _HAS_TQDM:
        bar.close()

    save_potential_energy(U_all, x_data, y_data, z_data, tag)
    save_magnetic_field(B_all, x_data, y_data, z_data, tag)
    save_currents(track, pod_x_list, pod_y_list, pod_z_list, currents_list)

    return U_all


def energy_potential_rotated(pod, track, pod_pos, pod_pts, tag,
                             roll, pitch, yaw, cfg=DEFAULT_SIM_CONFIG):
    # added by Wilf
    """
    energy_potential_static with the pod held at a fixed tilt
    (roll, pitch, yaw) in radians: fixed currents, but the pod sample
    points and dipole moment are rotated about pod_pos before each grid
    shift. Reproduces the dissertation's Fig 15/16 (saddle-point shift
    under tilt), which uses the static field. Args/return as
    energy_potential_static.
    """
    x_data, y_data, z_data = _make_grid(cfg)
    nx, ny, nz = cfg.x_num, cfg.y_num, cfg.z_num

    U_all = np.zeros((nx, ny, nz))
    B_all = [np.zeros((nx, ny, nz)) for _ in range(3)]

    R = rotation_matrix_from_euler(roll, pitch, yaw)
    m_vec = R @ dipole_moment(pod, pod_pts)
    pod_pts_body = (pod_pts - pod_pos) @ R.T  # tilted pod-frame offsets

    assign_currents_for_pod(track, pod_xy=0.0,
                            pod_total_mass=pod.TotalMass, config=cfg)

    total = nx * ny * nz
    bar   = _progress(range(nz), total=total, desc="Rotated potential")

    for k, z in enumerate(z_data):
        grav_term = pod.TotalMass * 9.81 * z
        for i, x in enumerate(x_data):
            for j, y in enumerate(y_data):
                target      = np.array([x, y, z])
                pts_shifted = pod_pts_body + target

                if _any_inside(track, pts_shifted):
                    U_all[i, j, k] = np.nan
                else:
                    U_all[i, j, k] = grav_term
                    for pt in pts_shifted:
                        Bx, By, Bz = track_field_at_point(track, pt)
                        B_all[0][i, j, k] += Bx
                        B_all[1][i, j, k] += By
                        B_all[2][i, j, k] += Bz
                        U_all[i, j, k]    -= (m_vec[0]*Bx
                                              + m_vec[1]*By
                                              + m_vec[2]*Bz)
        if _HAS_TQDM:
            bar.update(ny * nx)

    if _HAS_TQDM:
        bar.close()

    save_potential_energy(U_all, x_data, y_data, z_data, tag)
    save_magnetic_field(B_all, x_data, y_data, z_data, tag)

    return U_all


def energy_potential_ideal(pod, pod_pos, pod_pts, tag,
                           cfg=DEFAULT_SIM_CONFIG):
    """
    Potential energy sweep using the analytic ideal field
    (B_field_valley + B_field_gravity), no track needed. pod_pos is the
    (3,) reference pod position, pod_pts the (N, 3) sample points, tag
    the RunTag. Returns a (x_num, y_num, z_num) array.
    """
    x_data, y_data, z_data = _make_grid(cfg)
    nx, ny, nz = cfg.x_num, cfg.y_num, cfg.z_num

    U_all = np.zeros((nx, ny, nz))
    B_all = [np.zeros((nx, ny, nz)) for _ in range(3)]
    m_vec = dipole_moment(pod, pod_pts)
    n_pts = pod_pts.shape[0]

    total = nx * ny * nz
    bar   = _progress(range(nz), total=total, desc="Ideal potential")

    for k, z in enumerate(z_data):
        grav_term = pod.TotalMass * 9.81 * z
        for i, x in enumerate(x_data):
            for j, y in enumerate(y_data):
                shift       = np.array([x - pod_pos[0],
                                        y - pod_pos[1],
                                        z - pod_pos[2]])
                pts_shifted = pod_pts + shift

                U_all[i, j, k] = grav_term
                for pt in pts_shifted:
                    B = B_field_valley(pt) + B_field_gravity(pt, pod.TotalMass, n_pts)
                    B_all[0][i, j, k] = B[0]
                    B_all[1][i, j, k] = B[1]
                    B_all[2][i, j, k] = B[2]
                    U_all[i, j, k]   += abs(np.dot(m_vec, B))

        if _HAS_TQDM:
            bar.update(ny * nx)

    if _HAS_TQDM:
        bar.close()

    save_potential_energy(U_all, x_data, y_data, z_data, tag)
    save_magnetic_field(B_all, x_data, y_data, z_data, tag)

    return U_all
