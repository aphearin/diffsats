"""
"""
import numpy as np
from jax import numpy as jnp
from jax.experimental.ode import odeint as jodeint

from .. import nfw
from SatGen import config as cfg
from SatGen import profiles as pr
from SatGen.orbit import orbit


def test_nfw():
    """
    Enforce that the SatGen NFW profile and the diffsats NFW profile
    returns the same density.
    """
    M, conc = 1e12, 5.0
    h = pr.NFW(
        M,
        conc,
    )
    rs = h.rs
    Deltah = h.Deltah
    redshift = h.z
    littleh = cfg.h
    Om = cfg.Om
    OL = cfg.OL
    R = 100.0
    rho_diffsats = nfw.density(R, 0.0, conc, rs, Deltah, redshift, littleh, Om, OL)
    rho_SatGen = h.rho(R)
    assert abs(rho_diffsats - rho_SatGen) / rho_SatGen < cfg.eps


def test_orbit():
    """
    Enforce that a test particle within a SatGen NFW profile and the
    diffsats NFW profile evolve along the same orbit.
    """
    # ---define host potential for SatGen
    M, conc = 1e12, 5.0
    h = pr.NFW(M, conc)
    rs = h.rs
    Deltah = h.Deltah
    redshift = h.z
    littleh = cfg.h
    Om = cfg.Om
    OL = cfg.OL

    # ---initialize and integrate orbit with SatGen
    R = 100.0
    Vphi = h.Vcirc(R)
    xv = np.array([R, 0.0, 0.0, 0.0, Vphi, 0.0])
    o = orbit(xv)
    t = np.array([0.0, 2.0])  # [Gyr]
    m = 1e11  # satellite mass
    cfg.lnL_type = 4  # <<< use constant Coulomb log of lnL = 3.
    cfg.lnL_pref = 1.0  # <<< multiplier for Coulomb log
    o.integrate(t, h, m)
    # o.integrate(t,h,m=None) # <<< no dyn fric
    R_SatGen = o.xv[0]
    phi_SatGen = o.xv[1]

    # ---integrate the same orbit with diffsats
    xv = jnp.array([R, 0.0, 0.0, 0.0, Vphi, 0.0])
    t = jnp.array([0.0, 2.0])  # [Gyr]
    args = (m, conc, rs, Deltah, redshift, littleh, Om, OL)
    xvArray = jodeint(nfw.rhs_orbit_ode, xv, t, *args)
    R_diffsats = xvArray[-1][0]
    phi_diffsats = xvArray[-1][1]

    assert abs(R_diffsats - R_SatGen) / R_SatGen < cfg.eps
    assert abs(phi_diffsats - phi_SatGen) / phi_SatGen < cfg.eps
