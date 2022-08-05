import jax.numpy as jnp
from jax import jit as jjit
import numpy as np

from SatGen import orbit
from SatGen import config as cfg
from SatGen import cosmo as co
from SatGen.profiles import NFW, Vcirc


@jjit
def cylindrical_to_cartesian(cyl):
    """
    Converting cylindrical coordinates to cartesian.

    Can be used with vmap to input multiple
    coordinate values.

    Inputs:
        jnp.array((r,phi,z))
    Outputs:
        jnp.array((x,y,z))

    """
    r, phi, z = cyl
    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    z = z
    return jnp.array((x, y, z))


@jjit
def cartesian_to_cylindrical(cart):

    """
    Converting cartesian coordinates to cylindrical.

    Can be used with vmap to input multiple
    coordinate values.

    Inputs:
        jnp.array((x,y,z))
    Outputs:
        jnp.array((r,phi,z))

    """

    x, y, z = cart

    r = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)
    z = z
    return jnp.array((r, phi, z))


@jjit
def scalar_dot_prod(posvel_cylindrical):

    """
    Takes the cartesian scalar dot product to obtain
    the cos(theta) value between a coordinate vector
    and a velocity vector.

    Inputs:
        xvArray as (r, phi, z, Vr, Vphi, Vz)
    Outputs:
        cos(theta)

    """

    r, phi, z, Vr, Vphi, Vz = posvel_cylindrical

    x, y, z = cylindrical_to_cartesian(posvel_cylindrical[0:3])
    Vx, Vy, Vz = cylindrical_to_cartesian(posvel_cylindrical[3:])

    vec_pos = jnp.array((x, y, z))
    vec_vel = jnp.array((Vx, Vy, Vz))

    return jnp.dot(vec_pos, vec_vel)


@jjit
def get_normalized_posvel(posvel_cylindrical, rvir, desired_speed):

    """Accept and return 6D coordinate after normalizing position
    by rvir and speed by a desired speed (ex: Vcirc/3).

    Goes from cylindrical -> cartesian -> cylindridal"""

    # --- Import cylindrical 6D coord:
    pos_cyl = posvel_cylindrical[0:3]
    vel_cyl = posvel_cylindrical[3:]

    # --- Convert to cartesian:
    pos_cart = cylindrical_to_cartesian(pos_cyl)
    vel_cart = cylindrical_to_cartesian(vel_cyl)

    norm_pos_cart = jnp.dot(pos_cart, pos_cart)
    norm_vel_cart = jnp.dot(vel_cart, vel_cart)

    pos_cart_out = pos_cart * (rvir / norm_pos_cart)
    vel_cart_out = vel_cart * (desired_speed / norm_vel_cart)

    # --- Convert back to cylindrical:
    pos_cyl_out = cartesian_to_cylindrical(pos_cart_out)
    vel_cyl_out = cartesian_to_cylindrical(vel_cart_out)

    pos_cyl_out_tot = jnp.array((*pos_cyl_out, *vel_cyl_out))

    return pos_cyl_out_tot


def satgen_noDF_theta(Mratio, cNFW_, r0, phi0, z0, Vr0, Vphi0, Vz0):

    """
    SatGen without dynamical friction to calculate the 3D position
    values to be used to obtain the pericentric distance and time,
    and cos(theta) values.

    Inputs:
        Mratio: mass ratio of the subhalo and host (host=1e13Msun)
        cNFW_: default = 5
        r0: initial r-coordinate (cylindrical)
        phi0: initial phi-coordinate (cylindrical)
        z0: initial z-coordinate (cylindrical)
        Vr0: initial velocity in r-direction
        Vphi0: initial velocity in phi-direction
        Vz0: initial velocity in z-direction
    Outputs:
        rho: 3D distance [array]
        theta: cos(theta) value from dot product
    """

    # --- Host Properties (same as in test_profiles.py)
    Mv = 1e13
    cfg.lnL_type = 4

    # --- Defining the Profiles
    Delta_BN = co.DeltaBN(z=0, Om=0.3, OL=0.7)
    hNFW = NFW(Mv, cNFW_, Delta=Delta_BN)
    potential = [hNFW]

    # --- Initial Orbit Control (for mass evolution, evolve.py)
    rvir0 = hNFW.rh  # virial radius 557.8349
    Vcirc0 = Vcirc(potential, rvir0, z0) * 0.5  # 141.98784

    xv0 = jnp.array((r0, phi0, z0, Vr0, Vphi0, Vz0))
    # --- For Evolution:
    Nstep = 199  # number of timesteps
    timesteps = np.linspace(0.0, 10, Nstep)[1::]  # [Gyr]

    oNFWnoDF = orbit(xv0)

    oNFWnoDF.integrate(timesteps, potential, m=None)  # set to None for no DF
    ro = oNFWnoDF.xvArray[:, 0]
    zo = oNFWnoDF.xvArray[:, 2]

    rho = jnp.sqrt(ro**2 + zo**2)
    lastidx = len(rho) - 1
    rho[:lastidx]

    posvel = xv0
    norm_coords = get_normalized_posvel(posvel, rvir0, Vcirc0 / 3)  # returns cyl coords
    theta = scalar_dot_prod(norm_coords)  # gives cos(theta) from dot product

    return rho, theta


# --- Functions below are to determine the pericentric distance (and time).


def d_peri_finder(list_name):

    """
    Computes the pericentric distance as local minimum values
    from the 3D distance (rho).
    """
    return [
        minval
        for i, minval in enumerate(list_name)
        if ((i == 0) or (list_name[i - 1] >= minval))
        and ((i == len(list_name) - 1) or (minval < list_name[i + 1]))
    ]


def d_t_peri_finder(list_name, time_array):
    """
    Getting values for d_peri by finding values such that there is a
    higher value of 3D distance on either side of it, or such that
    the value is the lowest value with a higher value on one side,
    and no other/lower value on it's other side.

    Also computes pericentric time by index matching.
    """

    # --- Input list_name needs to be of type 'list'

    d_peri_vals = [
        minval
        for i, minval in enumerate(list_name)
        if ((i == 0) or (list_name[i - 1] >= minval))
        and ((i == len(list_name) - 1) or (minval < list_name[i + 1]))
    ]

    # --- Using values of d_peri to obtain t_peri, by index matching.
    # --- Getting d_peri values from d_peri_finder

    d_peri_vals = d_peri_finder(list_name)

    # --- Getting indeces of d_peri_finder values
    t_idx_vals = []
    for i in d_peri_vals:
        idx = list_name.index(i)
        t_idx_vals.append(idx)

    # --- Matching indices with time_array indices
    t_peri_ = []
    for j in t_idx_vals:
        t_peri_vals = time_array[j]
        t_peri_.append(t_peri_vals)

    return ("d_peri [kpc]:", d_peri_vals, "t_peri [Gyr]:", t_peri_)
