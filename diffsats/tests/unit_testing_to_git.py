#!/usr/bin/env python
# coding: utf-8
# 
# These functions are used as a unit test on R and phi 
# orbital values in the xvArray.

# In[1]:


import numpy as np
from jax import numpy as jnp
from jax.experimental.ode import odeint as jodeint


# In[2]:


import sys
import os

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path+"\\SatGen")
sys.path.append("/Users/naomi/Desktop/Work/repositories/python/SatGen")
import SatGen
from SatGen import orbit
from SatGen.orbit import orbit
from SatGen import evolve
from SatGen import config as cfg
from SatGen import cosmo as co
from SatGen import profiles as pr
from SatGen.profiles import NFW, Dekel, Einasto, MN, Vcirc, ftot, fDF
import time 
import re


# In[3]:


from diffsats import nfw


# In[4]:


def check_assert(value):

    '''
    Convert None in assert.allclose() to int 0. 
    Else return value.
    '''

    if value == None:
        return 0
    return value


# In[5]:


def check_assert_vals(lst):
    
    '''
    Check to make sure that all values in check_assert
    function are the same (all zero). If not, return 
    an error.
    '''
    return np.unique(lst).shape[0]<=1


# In[6]:


def test_orbit_UNIT(Mhost_, mass_ratio, conc_,  R0_, phi0_max, 
                    z0_max, VR0_max, Vz0_max, tfinal_, steps):
    '''
    Make sure that test particle in SatGen and diffsats
    NFW profile have the same orbit.
    '''
    
    #===== SatGen host potential =====
    M, conc = Mhost_, conc_
    hNFW = pr.NFW(M, conc)
    rs = hNFW.rs
    Deltah = hNFW.Deltah
    redshift = hNFW.z
    littleh = cfg.h
    Om = cfg.Om
    OL = cfg.OL
    
    t = np.array([0.0, tfinal_]) # Gyr
    m = mass_ratio * M # subhalo mass
    cfg.lnL_type = 4 # Coulomb log of lnL=3
    cfg.lnL_pref = 1.0 # Colomb log multiplier
    
    #===== SatGen Orbit Integration =====
    
    R = np.full((steps), R0_)
    Vphi = np.full((steps), hNFW.Vcirc(R0_))
    phi0_vals = np.linspace(0, phi0_max, steps)
    z0_vals = np.linspace(0, z0_max, steps)
    VR0_vals = np.linspace(0, VR0_max, steps)
    Vz0_vals = np.linspace(0, Vz0_max, steps)

    #--- Looping over multiple i.c. values
    xv_SG = []
    for g in range(len(R)):
        for h in range(len(phi0_vals)):
            for i in range(len(z0_vals)):
                for j in range(len(VR0_vals)):
                    for k in range(len(Vphi)):
                        for l in range(len(Vz0_vals)):
                            xv_in_loop = np.array([R[g], phi0_vals[h], z0_vals[i], 
                                                   VR0_vals[j], Vphi[k], Vz0_vals[l]])
                            xv_SG.append(xv_in_loop)
    print('>>> Getting SATGEN xv(r, phi, z, vr, vphi, vz)... >>>')
    #for array in xv_SG:
    #    print(array)

    o_SG = []
    for i in range(len(xv_SG)):
        o = orbit(xv_SG[i])
        o_SG.append(o)
    print('>>> Getting orbit objects... >>>')
    
    oint_SG = []
    for i in range(len(o_SG)):
        o_SG[i].integrate(t, hNFW, m)
        oint_SG.append(o_SG[i])
    print('>>> Integrating orbits... >>>')
    # o.integrate(t,hNFW, m=None) # no DF
    R_SG = []
    phi_SG = []
    for i in range(len(oint_SG)):
        R_SatGen = oint_SG[i].xv[0]
        phi_SatGen = oint_SG[i].xv[1]
        
        R_SG.append(R_SatGen)
        phi_SG.append(phi_SatGen)
    print('>>> Getting SATGEN R and phi values... >>>')
    print('')

    #===== diffsats Orbit Integration =====
    
    #--- Looping over multiple i.c. values
    xv_DS = []
    for g in range(len(R)):
        for h in range(len(phi0_vals)):
            for i in range(len(z0_vals)):
                for j in range(len(VR0_vals)):
                    for k in range(len(Vphi)):
                        for l in range(len(Vz0_vals)):
                            xv_in_loop = jnp.array([R[g], phi0_vals[h], z0_vals[i], 
                                                   VR0_vals[j], Vphi[k], Vz0_vals[l]])
                            xv_DS.append(xv_in_loop)
    
    print('>>> Getting diffsats xv(r, phi, z, vr, vphi, vz)... >>>')    
    t = jnp.array([0.0, tfinal_]) # Gyr
    print('>>> Getting orbit objects... >>>')
    args = (m, conc, rs, Deltah, redshift, littleh, Om, OL)
    
    print('>>> Integrating orbits... >>> ')
    xvArray = []
    for i in range(len(xv_DS)):
        xvArray_vals = jodeint(nfw.rhs_orbit_ode, xv_DS[i], t, *args)
        xvArray.append(xvArray_vals)
  
    print('>>> Getting diffsats R and phi values... >>>')
    R_DS = []
    phi_DS = []   
    for i in range(len(xvArray)):
        R_diffsats = xvArray[i][-1][0]
        phi_diffsats = xvArray[i][-1][1]
        
        R_DS.append(R_diffsats)
        phi_DS.append(phi_diffsats)

    for i in range(len(R_DS)):
        for j in range(len(R_SG)):
            allclose_var_R = np.testing.assert_allclose(R_DS[i], R_SG[j], rtol=1e-5, atol=1e-5)
            var_R_list = check_assert(allclose_var_R) # takes None -> 0 
    print('Checking R values:', check_assert_vals(var_R_list))
            
            
    for i in range(len(phi_DS)):
        for j in range(len(phi_SG)):
            allclose_var_phi = np.testing.assert_allclose(phi_DS[i], phi_SG[j], rtol=1e-5, atol=1e-5)
            var_phi_list = check_assert(allclose_var_phi)
    print('Checking phi values:', check_assert_vals(var_phi_list))
    


# In[7]:


UNIT_test_orbit_NEW(Mhost_=1e13, mass_ratio=1e-2, conc_=5.0,  R0_=444, 
                phi0_max=0.0, z0_max=0.0, VR0_max=0.0, Vz0_max=0.0, tfinal_=10, steps=2)


# Takes quite a bit of time to get to a mass ratio above 0.06 for a set host mass of 1e13... and there's an error.
# 
# assert statements require agreement to working precision of floats. np.allclose(A,B) where A,B are arrays (checks each element, return True or False, can then use assert; can also use RTOL and ATOL here) or floats.
# 
# Change function name to start with 'test' when submitting to git. submit to test/testorbit (something like that)

# In[ ]:




