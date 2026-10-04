"""
Field evaluation on Magnet and Collection objects. This was the field_at()
methods on PermMagnet/ElectroMagnet plus Full_Field_At_Point and
Field_On_Pod_Point in baseline_code.py; it lives here so the Magnet
classes stay pure geometry.
"""

import numpy as np

from geometry.shapes import Magnet, PermMagnet, ElectroMagnet
from geometry.collection import Collection
from physics.field import B_calc


def field_at_points(magnet, points):
    """
    Field (Tesla) at one or more points from a single magnet, as an
    (N, 3) array. A PermMagnet uses its magnetic_strength directly; an
    ElectroMagnet gets an effective M from turns * current * mu_r / L.
    """
    points = np.atleast_2d(points)
    B      = np.zeros((points.shape[0], 3))

    a       = magnet.radius
    L       = magnet.height
    b       = L / 2.0
    zcentre = magnet.position[2]

    if isinstance(magnet, PermMagnet):
        M = magnet.magnetic_strength
        I = 0.0
    elif isinstance(magnet, ElectroMagnet):
        I  = magnet.current
        NI = magnet.turns * I * magnet.mu_r
        M  = NI / L
    else:
        raise TypeError(f"Unsupported magnet type: {type(magnet)}")

    for i, p in enumerate(points):
        x = p[0] - magnet.position[0]
        y = p[1] - magnet.position[1]
        z = p[2]

        Bx, By, Bz = B_calc(x=x, y=y, z=z, I=I, a=a, L=L,
                             M=M, zcentre=zcentre, b=b)
        B[i] = [Bx, By, Bz]

    return B


def field_from_collection(collection, points):
    """Superposed field (Tesla) at the given points from every magnet in
    a Collection, as an (N, 3) array."""
    points = np.atleast_2d(points)
    B      = np.zeros((points.shape[0], 3))

    for magnet in collection.Magnets:
        B += field_at_points(magnet, points)

    return B


def full_field_at_point(pod, track, point):
    """Combined (3,) field at one point from both the pod and the track."""
    point = np.atleast_2d(point)
    B     = np.zeros(3)
    B    += field_from_collection(pod,   point)[0]
    B    += field_from_collection(track, point)[0]
    return B


def track_field_at_point(track, point):
    """
    (3,) field at one point from the track only. Used in the potential
    energy sweeps, where the pod is a passive dipole rather than a source.
    """
    point = np.atleast_2d(point)
    return field_from_collection(track, point)[0]


def field_gradient_at_point(source, point, h=1e-5):
    # added by Wilf
    """
    Central-difference spatial gradient of the field, as a (3, 3) array
    with grad[i, j] = dB_i/dx_j in Tesla/m. source is a Magnet or a
    Collection, h the step in metres. Needed where the gradient has to be
    live: the allocation Jacobian and the closed-loop driver.
    """
    point = np.asarray(point, dtype=float)

    def B_at(p):
        if isinstance(source, Collection):
            return field_from_collection(source, p)[0]
        return field_at_points(source, p)[0]

    grad = np.zeros((3, 3))
    for j in range(3):
        offset      = np.zeros(3)
        offset[j]   = h
        grad[:, j]  = (B_at(point + offset) - B_at(point - offset)) / (2 * h)

    return grad
