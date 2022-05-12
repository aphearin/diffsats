"""Equations of orbital motion in an axisymmetric NFW potential

This JAX-based implementation mirrors SatGen as closely as possible.

"""
from jax import jit as jjit
from jax import numpy as jnp


NEWTON_G = 4.4985e-06  # gravitational constant [kpc^3 Gyr^-2 Msun^-1]
RHOC0 = 277.5  # [h^2 Msun kpc^-3]


@jjit
def rho_crit(z, h, Om, OL):
    return RHOC0 * h**2 * (Om * (1.0 + z) ** 3 + OL)


@jjit
def _nfw_f(x):
    one_plus_x = 1.0 + x
    return jnp.log(one_plus_x) - x / one_plus_x


@jjit
def _get_phi0(conc, rs, Deltah, redshift, littleh, Om, OL):
    rhoc = rho_crit(redshift, littleh, Om, OL)
    rho0 = rhoc * Deltah / 3.0 * conc**3.0 / _nfw_f(conc)
    phi0 = -4.0 * jnp.pi * NEWTON_G * rho0 * rs**2.0
    return phi0


@jjit
def grav_accel(R, z, conc, rs, Deltah, redshift, littleh, Om, OL):
    """Gravitational acceleration in axisymmetric NFW potential at location (R, z)

    Parameters
    ----------
    R : float
        xy distance from z-axis in kpc

    z : float
        z-coordinate in kpc

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
def sigma(R,z,...):
    """
    NFW.sigma
    """
    pass

@jjit
def rho(R,z,...):
    """
    NFW.rho
    """
    pass


@jjit
def df_accel(xv,m,Mh, Deltah, redshift, littleh, Om, OL):
    """
    Dynamical-friction (DF) acceleration [(kpc/Gyr)^2 kpc^-1] given 
    satellite mass, phase-space coordinate, and axisymmetric host
    potential:
    
        f_DF = -4piG^2 m Sum_i rho_i(R,z)F(<|V_i|)ln(Lambda_i)V_i/|V_i|^3  
    
    where
        
        V_i: relative velocity (vector) of the satellite with respect to 
            the host component i
        F(<|V_i|) = erf(X) - 2X/sqrt{pi} exp(-X^2) with 
            X = |V_i| / (sqrt{2} sigma(R,z))
        ln(Lambda_i): Coulomb log of host component i 

    """
    R, phi, z, VR, Vphi, Vz = xv
    VrelR = VR
    Vrelphi = Vphi
    Vrelz = Vz
    Vrel = np.sqrt(VrelR**2.+Vrelphi**2.+Vrelz**2.)
    Vrel = max(Vrel,cfg.eps) # safety
    lnL = np.log(Mh/m) # Coulomb logarithm
    X = Vrel / (cfg.Root2 * sigma(R,z, ...))
    fac_s = rho(R,z) * lnL * ( erf(X) - cfg.TwoOverRootPi * X*np.exp(-X**2.) ) / Vrel**3 
    fac = -cfg.FourPiGsqr * m # common factor in f_DF 
    fR = fac*fac_s * VrelR
    fphi = fac*fac_s * Vrelphi
    fz = fac*fac_s * Vrelz
    return fR, fphi, fz

@jjit
def rhs_orbit_ode(t, y,  conc, rs, Deltah, redshift, littleh, Om, OL):
    """
    Returns right-hand-side functions of the EOMs for orbit integration:

        d R / d t = VR
        d phi / d t = Vphi / R
        d z / d t = Vz
        d VR / dt = Vphi^2 / R + fR
        d Vphi / dt = - VR * Vphi / R + fphi
        d Vz / d t = fz

    """
    R, phi, z, VR, Vphi, Vz = y
    R = max(R, 1e-6)
    # Not sure yet how to handle the additional arguments accepted by grav_accel
    fR, fphi, fz = grav_accel(R, z, conc, rs, Deltah, redshift, littleh, Om, OL)
    return VR, Vphi / R, Vz, Vphi**2.0 / R + fR, -VR * Vphi / R + fphi, fz

