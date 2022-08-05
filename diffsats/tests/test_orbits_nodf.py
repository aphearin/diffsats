"""
"""


def test_orbit_UNIT_noDF(
    Mhost_, mass_ratio, conc_, phi0_max, z0_max, VR0_max, Vz0_max, tfinal_, steps
):
    """
    Make sure that test particle in SatGen and diffsats
    NFW profile have the same orbit without dynamical friction.

    Note that here diffsats doesn't use the dynamical friction
    component when calculating the total acceleration
    (only taking the gravitational acceleration into account).

    A new function was added to nfw.py to include a noDF case,
    called rhs_orbit_ode_noDF.

    Uses RK solver.
    """

    # ===== SatGen host potential =====
    M, conc = Mhost_, conc_
    Delta_BN = co.DeltaBN(z=0, Om=0.3, OL=0.7)
    hNFW = NFW(M, conc, Delta=Delta_BN)
    potential = [hNFW]
    rs = hNFW.rs
    redshift = 0.0
    littleh = cfg.h
    Om = cfg.Om
    OL = cfg.OL

    t = np.array([0.0, tfinal_])  # Gyr
    m = mass_ratio * M  # subhalo mass
    cfg.lnL_type = 4  # Coulomb log of lnL=3
    # cfg.lnL_pref = 1.0 # Colomb log multiplier

    # ===== SatGen Orbit Integration =====

    R = np.full((1), hNFW.rh)
    # Vphi = np.full((steps), hNFW.Vcirc(R)*0.5)
    phi0_vals = np.linspace(0, phi0_max, steps)
    z0_vals = np.linspace(0, z0_max, steps)
    VR0_vals = np.linspace(0, VR0_max, steps)
    Vphi = np.full((1), Vcirc(potential, hNFW.rh, z0_max) * 0.5)
    Vz0_vals = np.linspace(0, Vz0_max, steps)

    # --- Looping over multiple i.c. values
    xv_SG_noDF = []
    for g in range(len(R)):
        for h in range(len(phi0_vals)):
            for i in range(len(z0_vals)):
                for j in range(len(VR0_vals)):
                    for k in range(len(Vphi)):
                        for l in range(len(Vz0_vals)):
                            xv_in_loop = np.array(
                                [
                                    R[g],
                                    phi0_vals[h],
                                    z0_vals[i],
                                    VR0_vals[j],
                                    Vphi[k],
                                    Vz0_vals[l],
                                ]
                            )
                            xv_SG_noDF.append(xv_in_loop)

    o_SG_noDF = []
    for i in range(len(xv_SG_noDF)):
        o_noDF = orbit(xv_SG_noDF[i])
        o_SG_noDF.append(o_noDF)

    oint_SG_noDF = []
    for i in range(len(o_SG_noDF)):
        o_SG_noDF[i].integrate(t, potential, m=None)  # no DF if m=None
        oint_SG_noDF.append(o_SG_noDF[i])

    R_SG_noDF = []
    phi_SG_noDF = []
    for i in range(len(oint_SG_noDF)):
        R_SatGen = oint_SG_noDF[i].xv[0]
        phi_SatGen = oint_SG_noDF[i].xv[1]

        R_SG_noDF.append(R_SatGen)
        phi_SG_noDF.append(phi_SatGen)

    # ===== diffsats Orbit Integration =====

    # --- Looping over multiple i.c. values
    mdef_string = str(int(Delta_BN)) + "c"
    ht_nfw = NFWPhaseSpace(mdef=mdef_string, redshift=0.0)
    Mhalo_h = Mhost_ * littleh
    Rhalo_mpc_h = ht_nfw.halo_mass_to_halo_radius(Mhalo_h)
    Rhalo_kpc = Rhalo_mpc_h * 1000 / littleh

    rs_kpc = Rhalo_kpc / conc_
    R0_DS = Rhalo_kpc

    km_s_to_kpc_gyr = 1.0227
    Vphi_DS = np.full(
        (steps),
        ht_nfw.circular_velocity(Rhalo_mpc_h, Mhalo_h, conc_) * 0.5 * km_s_to_kpc_gyr,
    )

    xv_DS_noDF = []

    for g in range(len(R)):
        for h in range(len(phi0_vals)):
            for i in range(len(z0_vals)):
                for j in range(len(VR0_vals)):
                    for k in range(len(Vphi)):
                        for l in range(len(Vz0_vals)):
                            xv_in_loop = jnp.array(
                                [
                                    R[g],
                                    phi0_vals[h],
                                    z0_vals[i],
                                    VR0_vals[j],
                                    Vphi_DS[k],
                                    Vz0_vals[l],
                                ]
                            )
                            xv_DS_noDF.append(xv_in_loop)

    t = jnp.array([0.0, tfinal_])  # Gyr
    args = (m, conc, rs_kpc, Delta_BN, redshift, littleh, Om, OL)

    xvArray = []
    for i in range(len(xv_DS_noDF)):
        xvArray_vals = jodeint(nfw.rhs_orbit_ode_noDF, xv_DS_noDF[i], t, *args)
        xvArray.append(xvArray_vals)

    R_DS_noDF = []
    phi_DS_noDF = []
    for i in range(len(xvArray)):
        R_diffsats = xvArray[i][-1][0]
        phi_diffsats = xvArray[i][-1][1]

        R_DS_noDF.append(R_diffsats)
        phi_DS_noDF.append(phi_diffsats)

    idx_range = int(len(phi_DS_noDF) / steps)

    for i in range(len(R_DS_noDF)):
        for j in range(len(R_SG_noDF)):
            np.testing.assert_allclose(R_DS_noDF[i], R_SG_noDF[j], rtol=1e-2, atol=1e-2)

    for i in range(len(phi_DS_noDF[0:idx_range])):
        for j in range(len(phi_SG_noDF[0:idx_range])):
            np.testing.assert_allclose(
                phi_DS_noDF[i], phi_SG_noDF[j], rtol=1e-2, atol=1e-2
            )


# An example of a successful test.

# test_plt1 = test_orbit_UNIT_noDF(Mhost_=1e13, mass_ratio=1e-6, conc_=5.0,
#                     phi0_max=0.0, z0_max=0.0, VR0_max=0.0, Vz0_max=0.0, tfinal_=10,
#                                 steps=2)
