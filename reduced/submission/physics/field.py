"""
Low-level magnetic field kernels, working on raw scalars and arrays with
no Magnet or Collection objects. From baseline_code.py (roughly lines
746-892), unchanged bar importing MU_0 in place of the old global `mu`.

elliptic, B_rho, B_z and constants are originally Bella Mak's, with the
boundary-clamping tweaks noted inline; B_calc and closed_cyl are hers as
modified by Lewis.
"""

import numpy as np
import numba

from config import MU_0


# these numba helpers must stay at module level to compile at import time

@numba.njit
def constants(rho, z, a, M, b):
    """Pre-compute geometric constants for a cylindrical magnet field."""
    B_0    = (MU_0 * M) / np.pi
    zplus  = z + b
    zminus = z - b

    alphap  = a / np.sqrt(zplus  ** 2 + (rho + a) ** 2)
    alpham  = a / np.sqrt(zminus ** 2 + (rho + a) ** 2)
    betap   = zplus  / np.sqrt(zplus  ** 2 + (rho + a) ** 2)
    betam   = zminus / np.sqrt(zminus ** 2 + (rho + a) ** 2)
    gamma   = (a - rho) / (a + rho)
    gammasq = gamma ** 2
    kplus   = np.sqrt((zplus  ** 2 + (a - rho) ** 2) / (zplus  ** 2 + (a + rho) ** 2))
    kminus  = np.sqrt((zminus ** 2 + (a - rho) ** 2) / (zminus ** 2 + (a + rho) ** 2))

    return B_0, zplus, zminus, alphap, alpham, betap, betam, gamma, gammasq, kplus, kminus


@numba.njit
def elliptic(kc, p, c, s):
    """
    Bulirsch complete elliptic integral.
    Taken directly from Bella Mak's code; boundary clamping adjusted.
    """
    if kc == 0:
        return np.nan

    errtol = 1e-6
    k  = abs(kc)
    pp = p
    cc = c
    ss = s
    em = 1.0

    if p > 0:
        pp = np.sqrt(p)
        ss = s / pp
    else:
        f  = kc * kc
        q  = 1.0 - f
        g  = 1.0 - pp
        f  = f - pp
        q  = q * (ss - c * pp)
        pp = np.sqrt(f / g)
        cc = (c - ss) / g
        ss = -q / (g * g * pp) + cc * pp

    f  = cc
    cc = cc + ss / pp
    g  = k / pp
    ss = 2 * (ss + f * g)
    pp = g + pp
    g  = em
    em = k + em
    kk = k

    while abs(g - k) > g * errtol:
        k  = 2 * np.sqrt(kk)
        kk = k * em
        f  = cc
        cc = cc + ss / pp
        g  = kk / pp
        ss = 2 * (ss + f * g)
        pp = g + pp
        g  = em
        em = k + em

    # Boundary clamping (modified from original)
    if kc < 1e-6:
        kc = 1e-6
    elif kc > 1.0 - 1e-10:
        kc = 1.0 - 1e-10

    return (np.pi / 2) * (ss + cc * em) / (em * (em + pp))


@numba.njit
def B_rho(B0, alpha_p, k_plus, alpha_m, k_minus):
    """Radial (rho) component of the cylindrical field. From Bella Mak."""
    return B0 * (
        alpha_p * elliptic(k_plus,  1.0, 1.0, -1.0)
      - alpha_m * elliptic(k_minus, 1.0, 1.0, -1.0)
    )


@numba.njit
def B_z(B0, a, rho, beta_p, k_plus, gamma_sq, gamma, beta_m, k_minus):
    """Axial (z) component of the cylindrical field. From Bella Mak."""
    denominator = a + max(rho, 1e-15)
    return (B0 * a / denominator) * (
        beta_p * elliptic(k_plus,  gamma_sq, 1.0, gamma)
      - beta_m * elliptic(k_minus, gamma_sq, 1.0, gamma)
    )


def B_calc(x, y, z, I, a, L, M, zcentre, b):
    """
    Field (Bx, By, Bz) in Tesla at Cartesian (x, y, z) from one cylinder:
    convert to cylindrical, call the Bella Mak kernel, convert back.

    x, y are relative to the magnet centre, z is absolute. I is current
    in A (0 for a permanent magnet), a the radius, L the height, M the
    effective magnetisation (A/m), zcentre the magnet-centre z, b the
    half-height L/2.
    """
    rho     = np.sqrt(x ** 2 + y ** 2)
    azimuth = np.arctan2(y, x)
    zref    = z - zcentre

    B_0, zplus, zminus, alphap, alpham, betap, betam, gamma, gammasq, kplus, kminus = (
        constants(rho, zref, a, M, b)
    )

    Bz_val   = B_z(B_0, a, rho, betap,  kplus,  gammasq, gamma, betam,  kminus)
    Brho_val = B_rho(B_0, alphap, kplus, alpham, kminus)

    return np.cos(azimuth) * Brho_val, np.sin(azimuth) * Brho_val, Bz_val


def closed_cyl(x, y, z, pod):
    """
    Total field (Bx, By, Bz) in Tesla at (x, y, z) in metres from every
    magnet in a pod Collection (PermMagnet objects), by the closed-form
    cylindrical formula. Bella Mak's code, extended to loop over a whole
    Collection instead of a single magnet.
    """
    Bx_total = By_total = Bz_total = 0.0

    for magnet in pod.Magnets:
        x0, y0, z0 = magnet.position
        Mz = magnet.magnetic_strength

        dx  = x - x0
        dy  = y - y0
        rho = np.sqrt(dx ** 2 + dy ** 2)
        phi = np.arctan2(dy, dx) if rho > 1e-12 else 0.0

        zt = z - (z0 + magnet.height / 2)
        zb = z - (z0 - magnet.height / 2)
        a  = magnet.radius

        gamma    = (zt - zb) / 2
        gamma_sq = gamma ** 2
        B0       = MU_0 * Mz / np.pi

        alpha_p = zt / np.sqrt((a + rho) ** 2 + zt ** 2)
        alpha_m = zb / np.sqrt((a + rho) ** 2 + zb ** 2)

        beta_p  = (a**2 - rho**2 - zt**2) / ((a - rho)**2 + zt**2 + 1e-12)
        beta_m  = (a**2 - rho**2 - zb**2) / ((a - rho)**2 + zb**2 + 1e-12)

        k_plus  = np.sqrt(4 * a * rho / ((a + rho)**2 + zt**2))
        k_minus = np.sqrt(4 * a * rho / ((a + rho)**2 + zb**2))

        Bz_val   = B_z(B0, a, rho, beta_p, k_plus, gamma_sq, gamma, beta_m, k_minus)
        Brho_val = B_rho(B0, alpha_p, k_plus, alpha_m, k_minus)

        if rho < 2e-6:
            Bz_val = np.clip(Bz_val, -1.0, 0.5)
        if rho < 1e-6:
            Brho_val = 0.0

        Bx_total += Brho_val * np.cos(phi) if rho > 1e-12 else 0.0
        By_total += Brho_val * np.sin(phi) if rho > 1e-12 else 0.0
        Bz_total += Bz_val

    return Bx_total, By_total, Bz_total
