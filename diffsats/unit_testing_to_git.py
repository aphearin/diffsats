# ### Unit testing with Dynamical Friction
#
# Comparing SatGen DF to diffsats DF.


def test_orbit_UNIT_DF(
    Mhost_, mass_ratio, conc_, phi0_max, z0_max, VR0_max, Vz0_max, tfinal_, steps
):
    """
    Make sure that test particle in SatGen and diffsats
    NFW profile have the same orbit. This function tests
    with dynamical friction included.

    In nfw.py, this uses the rhs_orbit_ode function.

    Uses RK method.
    """

    # ===== SatGen host potential =====
    M, conc = Mhost_, conc_
    Delta_BN = co.DeltaBN(z=0, Om=0.3, OL=0.7)
    hNFW = NFW(M, conc, Delta=Delta_BN)
    potential = [hNFW]
    rs = hNFW.rs
    Deltah = hNFW.Deltah
    redshift = hNFW.z
    littleh = cfg.h
    Om = cfg.Om
    OL = cfg.OL

    t = np.array([0.0, tfinal_])  # Gyr
    m = mass_ratio * M  # subhalo mass
    cfg.lnL_type = 4  # Coulomb log of lnL=3
    cfg.lnL_pref = 1.0  # Colomb log multiplier

    # ===== SatGen Orbit Integration =====

    R = np.full((1), hNFW.rh)
    # Vphi = np.full((1), Vcirc(potential, hNFW.rh, z0_max)*0.5)
    Vphi = np.full((1), Vcirc(potential, hNFW.rh) * 0.5)
    phi0_vals = np.linspace(0, phi0_max, steps)
    z0_vals = np.linspace(0, z0_max, steps)
    VR0_vals = np.linspace(0, VR0_max, steps)
    Vz0_vals = np.linspace(0, Vz0_max, steps)

    # --- Looping over multiple i.c. values
    xv_SG = []
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
                            xv_SG.append(xv_in_loop)

    o_SG = []
    for i in range(len(xv_SG)):
        o = orbit(xv_SG[i])
        o_SG.append(o)

    oint_SG = []
    for i in range(len(o_SG)):
        o_SG[i].integrate(t, hNFW, m)
        oint_SG.append(o_SG[i])

    R_SG = []
    phi_SG = []
    for i in range(len(oint_SG)):
        R_SatGen = oint_SG[i].xv[0]
        phi_SatGen = oint_SG[i].xv[1]

        R_SG.append(R_SatGen)
        phi_SG.append(phi_SatGen)

    # ===== diffsats Orbit Integration =====

    # --- Looping over multiple i.c. values
    xv_DS = []
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
                                    Vphi[k],
                                    Vz0_vals[l],
                                ]
                            )
                            xv_DS.append(xv_in_loop)

    t = jnp.array([0.0, tfinal_])  # Gyr
    args = (m, conc, rs, Deltah, redshift, littleh, Om, OL)

    xvArray = []
    for i in range(len(xv_DS)):
        xvArray_vals = jodeint(nfw.rhs_orbit_ode, xv_DS[i], t, *args)
        xvArray.append(xvArray_vals)

    R_DS = []
    phi_DS = []
    for i in range(len(xvArray)):
        R_diffsats = xvArray[i][-1][0]
        phi_diffsats = xvArray[i][-1][1]

        R_DS.append(R_diffsats)
        phi_DS.append(phi_diffsats)

    idx_range = int(len(phi_DS) / steps)

    for i in range(len(R_DS)):
        for j in range(len(R_SG)):
            np.testing.assert_allclose(R_DS[i], R_SG[j], rtol=1e-2, atol=1e-2)

    for i in range(len(phi_DS[0:idx_range])):
        for j in range(len(phi_SG[0:idx_range])):
            np.testing.assert_allclose(phi_DS[i], phi_SG[j], rtol=1e-2, atol=1e-2)


# Below is an example of a successful test.

# test_orbit_UNIT_DF(Mhost_=1e13, mass_ratio=1e-2, conc_=5.0,
#                phi0_max=0.0, z0_max=0.0, VR0_max=0.0, Vz0_max=0.0, tfinal_=10, steps=2)
