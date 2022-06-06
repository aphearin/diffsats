"""Equations of orbital motion in an axisymmetric NFW potential

This JAX-based implementation mirrors SatGen as closely as possible.

"""
from jax import jit as jjit
from jax import numpy as jnp
from jax.scipy.special import erf


NEWTON_G = 4.4985e-06  # gravitational constant [kpc^3 Gyr^-2 Msun^-1]
RHOC0 = 277.5  # [h^2 Msun kpc^-3]
EPS = 0.001  # an infinitesimal for various purposes


@jjit
def rho_crit(z, littleh, Om, OL):
    return RHOC0 * littleh**2 * (Om * (1.0 + z) ** 3 + OL)


@jjit
def _nfw_f(x):
    one_plus_x = 1.0 + x
    return jnp.log(one_plus_x) - x / one_plus_x


@jjit
def _get_rho0(conc, rs, Deltah, redshift, littleh, Om, OL):
    rhoc = rho_crit(redshift, littleh, Om, OL)
    rho0 = rhoc * Deltah / 3.0 * conc**3.0 / _nfw_f(conc)
    return rho0


@jjit
def _get_phi0(conc, rs, Deltah, redshift, littleh, Om, OL):
    rho0 = _get_rho0(conc, rs, Deltah, redshift, littleh, Om, OL)
    phi0 = -4.0 * jnp.pi * NEWTON_G * rho0 * rs**2.0
    return phi0


@jjit
def _get_rmax(rs):
    return 2.163 * rs


@jjit
def circ_vel(R, z, conc, rs, Deltah, redshift, littleh, Om, OL):
    """Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2)

    Parameters
    ----------
    R : float
        xy distance from z-axis in kpc

    z : float
        z-coordinate in kpc

    conc : float
        NFW concentration

    rs : float
        NFW scale radius in kpc

    Deltah : float
        Overdensity threshold defining the halo boundary

    redshift : float

    littleh : float
        Hubble parameter

    Om : float
        Omega matter

    OL : float
        Omega Lambda

    Returns
    -------
    vcirc : float
        Circular velocity in km/Gyr

    """
    r = jnp.sqrt(R**2.0 + z**2.0)
    return jnp.sqrt(
        r * -grav_accel(r, 0.0, conc, rs, Deltah, redshift, littleh, Om, OL)[0]
    )


@jjit
def grav_accel(R, z, conc, rs, Deltah, redshift, littleh, Om, OL):
    """Gravitational acceleration in axisymmetric NFW potential at location (R, z)

    Parameters
    ----------
    R : float
        xy distance from z-axis in kpc

    z : float
        z-coordinate in kpc

    conc : float
        NFW concentration

    rs : float
        NFW scale radius in kpc

    Deltah : float
        Overdensity threshold defining the halo boundary

    redshift : float

    littleh : float
        Hubble parameter

    Om : float
        Omega matter

    OL : float
        Omega Lambda

    Returns
    -------
    gravitational acceleration : 3-element tuple
        R-, phi-, z-components of the acceleration in units [(kpc/Gyr)^2 kpc^-1]
        [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]

    """
    r = jnp.sqrt(R**2.0 + z**2.0)
    x = r / rs
    phi0 = _get_phi0(conc, rs, Deltah, redshift, littleh, Om, OL)
    fac = phi0 * (_nfw_f(x) / x) / r**2.0
    fR, fphi, fz = fac * R, fac * 0.0, fac * z
    return fR, fphi, fz


@jjit
def vel_disp(R, z, conc, rs, Deltah, redshift, littleh, Om, OL):
    """Velocity dispersion [kpc/Gyr] at radius r = sqrt(R^2 + z^2)

    Parameters
    ----------
    R : float
        xy distance from z-axis in kpc

    z : float
        z-coordinate in kpc

    conc : float
        NFW concentration

    rs : float
        NFW scale radius in kpc

    Deltah : float
        Overdensity threshold defining the halo boundary

    redshift : float

    littleh : float
        Hubble parameter

    Om : float
        Omega matter

    OL : float
        Omega Lambda

    Returns
    -------
    vel_disp : float
        Velocity dispersion in units of kpc/Gyr

    Notes
    -----
    Assumes isotropic velicity dispersion tensor, following the
    Zentner & Bullock (2003) fitting function.

    """
    r = jnp.sqrt(R**2.0 + z**2.0)
    x = r / rs
    rmax = _get_rmax(rs)
    Vmax = circ_vel(rmax, 0.0, conc, rs, Deltah, redshift, littleh, Om, OL)
    return Vmax * 1.4393 * x**0.354 / (1.0 + 1.1756 * x**0.725)


@jjit
def density(R, z, conc, rs, Deltah, redshift, littleh, Om, OL):
    """Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2).

    Parameters
    ----------
    R : float
        xy distance from z-axis in kpc

    z : float
        z-coordinate in kpc

    conc : float
        NFW concentration

    rs : float
        NFW scale radius in kpc

    Deltah : float
        Overdensity threshold defining the halo boundary

    redshift : float

    littleh : float
        Hubble parameter

    Om : float
        Omega matter

    OL : float
        Omega Lambda

    Returns
    -------
    rho : float
        Physical density in units of Msun/kpc^3

    """
    r = jnp.sqrt(R**2.0 + z**2.0)
    x = r / rs
    rho0 = _get_rho0(conc, rs, Deltah, redshift, littleh, Om, OL)
    return rho0 / (x * (1.0 + x) ** 2.0)


@jjit
def df_accel(xv, m, conc, rs, Deltah, redshift, littleh, Om, OL):
    """Dynamical-friction (DF) acceleration [(kpc/Gyr)^2 kpc^-1]

    Parameters
    ----------
    xv : 6-element sequence
        phase-space coordinates in a cylindrical frame
        xv = (R, phi, z, VR, Vphi, Vz)
        units: [kpc, radian, kpc, kpc/Gyr, kpc/Gyr, kpc/Gyr]

    m : float
        Satellite mass in units of Msun

    conc : float
        NFW concentration

    rs : float
        NFW scale radius in kpc

    Deltah : float
        Overdensity threshold defining the halo boundary

    redshift : float

    littleh : float
        Hubble parameter

    Om : float
        Omega matter

    OL : float
        Omega Lambda

    Notes
    -----
        f_df = -4piG^2 m rho(R,z) F(<|V|) lnL V/|V|^3

    where

        V: relative velocity (vector) of the satellite with respect to host component i
        F(<|V|) = erf(X) - 2X/sqrt{pi} exp(-X^2) with
            X = |V| / (sqrt{2} sigma(R,z))
        lnL: Coulomb log

    """
    R, phi, z, VR, Vphi, Vz = xv
    VrelR = VR
    Vrelphi = Vphi
    Vrelz = Vz
    Vrel = jnp.sqrt(VrelR**2.0 + Vrelphi**2.0 + Vrelz**2.0)
    # Vrel = max(Vrel,EPS) # safety <<< jax complains about it...
    lnL = 3.0  # <<< to be updated, for testing, just use constant Coulomb logarithm
    X = Vrel / (
        jnp.sqrt(2.0) * vel_disp(R, z, conc, rs, Deltah, redshift, littleh, Om, OL)
    )
    rho = density(R, z, conc, rs, Deltah, redshift, littleh, Om, OL)
    fac_s = (
        rho
        * lnL
        * (erf(X) - 2.0 / jnp.sqrt(jnp.pi) * X * jnp.exp(-(X**2.0)))
        / Vrel**3
    )
    fac = -4.0 * jnp.pi * NEWTON_G**2.0 * m  # common pre factor
    fR = fac * fac_s * VrelR
    fphi = fac * fac_s * Vrelphi
    fz = fac * fac_s * Vrelz
    return fR, fphi, fz


@jjit
def rhs_orbit_ode(y, t, m, conc, rs, Deltah, redshift, littleh, Om, OL):
    """
    Returns right-hand-side functions of the EOMs for orbit integration:

        d R / d t = VR
        d phi / d t = Vphi / R
        d z / d t = Vz
        d VR / dt = Vphi^2 / R + fR
        d Vphi / dt = - VR * Vphi / R + fphi
        d Vz / d t = fz

    Parameters
    ----------
    y : 6-element sequence
        phase-space coordinates in a cylindrical frame
        y = (R, phi, z, VR, Vphi, Vz)
        units: [kpc, radian, kpc, kpc/Gyr, kpc/Gyr, kpc/Gyr]

    t : float
        Time in units of Gyr

    m : float
        Satellite mass in units of Msun

    conc : float
        NFW concentration

    rs : float
        NFW scale radius in kpc

    Deltah : float
        Overdensity threshold defining the halo boundary

    redshift : float

    littleh : float
        Hubble parameter

    Om : float
        Omega matter

    OL : float
        Omega Lambda

    Returns
    -------

    """
    R, phi, z, VR, Vphi, Vz = y

    fR_grav, fphi_grav, fz_grav = grav_accel(
        R, z, conc, rs, Deltah, redshift, littleh, Om, OL
    )
    fR_df, fphi_df, fz_df = df_accel(y, m, conc, rs, Deltah, redshift, littleh, Om, OL)
    fR = fR_grav + fR_df
    fphi = fphi_grav + fphi_df
    fz = fz_grav + fz_df
    return VR, Vphi / R, Vz, Vphi**2.0 / R + fR, -VR * Vphi / R + fphi, fz
