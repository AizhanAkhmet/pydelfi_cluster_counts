import numpy as np
import scipy.integrate as integrate
import pyccl as ccl

# Cosmology modulus
def cosmo(theta):
    # create a flat cosmology with massive neutrinos and some other default params
        
    # $\Omega_{m}h^{2} = \Omega_{cdm}h^{2} + \Omega_{b}h^{2}$, $\sigma_{8}$, $h$, $n_{s}$, $w_{0}$}
    Omega_cdm = theta[0]/(theta[3]**2)
    Omega_b = theta[1]/(theta[3]**2)
    sigma8 = theta[2]
    h = theta[3]
    n_s = theta[4]
    w0 = theta[5]
        
    # setting up different cosmology configuration to match benchmarks and keep CLASS from crashing.
    cosmo_ccl = ccl.Cosmology(Omega_c= Omega_cdm, Omega_b= Omega_b, sigma8= sigma8,
                                      h= h, n_s= n_s, w0= w0,
                                      Omega_g=0, Omega_k=0,
                                      Neff=0, m_nu=0.0,
                                      wa=0, T_CMB=2.7, transfer_function='bbks',mass_function='tinker')
        
    return cosmo_ccl

################ ~~~~~~~~~~~~~~~`START ~~~~~~~~~~~~~~~~~~~ #################
# return a halo mass object (at a given redshift z for a given range of masses)
def halo_mass_function_object(theta, masses, z):
    cosmo_ccl = cosmo(theta)
    scale_fact_a = 1/(1+z)
        
    # using Sheth et al. (1999) halo mass function - but we could extend it to other halo mass functions
    mass_funct = ccl.halos.hmfunc.MassFuncSheth99(cosmo_ccl)
        
    # calculate halo mass function for a given redshift and a given range of masses
    #dn_dM = mass_funct.get_mass_function(cosmo_ccl, masses, scale_fact_a)
        
    return mass_funct

################ ~~~~~~~~~~~~~~~~END  ~~~~~~~~~~~~~~~~~~~`~ ##################


# compute halo mass function (at a given redshift z for a given range of masses)
def halo_mass_function(theta, masses, z):
    cosmo_ccl = cosmo(theta)
    scale_fact_a = 1/(1+z)
        
    # using Sheth et al. (1999) halo mass function - but we could extend it to other halo mass functions
    mass_funct = ccl.halos.hmfunc.MassFuncSheth99(cosmo_ccl)
        
    # calculate halo mass function for a given redshift and a given range of masses
    dn_dM = mass_funct.get_mass_function(cosmo_ccl, masses, scale_fact_a)
        
    return dn_dM

# integrate halo mass function 
def n_counts_m_integral(log10mass, z, theta):
    cosmo_ccl = cosmo(theta)
    scale_fact_a = 1/(1+z)
    
    # using Sheth et al. (1999) halo mass function - but we could extend it to other halo mass functions
    mass_funct = ccl.halos.hmfunc.MassFuncSheth99(cosmo_ccl)
        
    # calculate halo mass function for a given redshift and a given range of masses
    dn_dlog10M = mass_funct.get_mass_function(cosmo_ccl, 10**log10mass, scale_fact_a)
    
    return dn_dlog10M

def n_counts_m_integral_only(z, log10mass_min, log10mass_max, theta):
    cosmo_ccl = cosmo(theta)
    #scale_fact_a = 1/(1+z)
    
    #H0 = 100*theta[3]
    #c = ccl.physical_constants.CLIGHT*1e-3 # in km/s
    #H_z = H0*ccl.background.h_over_h0(cosmo_ccl, scale_fact_a)
    
    # compute integral in m
    integral_m = integrate.quad(n_counts_m_integral, log10mass_min, log10mass_max, args=(z, theta))[0]
    
    return integral_m

def n_counts_full_integral(z, log10mass_min, log10mass_max, theta):
    cosmo_ccl = cosmo(theta)
    scale_fact_a = 1/(1+z)
    
    H0 = 100*theta[3]
    c = ccl.physical_constants.CLIGHT*1e-3 # in km/s
    H_z = H0*ccl.background.h_over_h0(cosmo_ccl, scale_fact_a)
    
    # compute integral in m
    integral_m = integrate.quad(n_counts_m_integral, log10mass_min, log10mass_max, args=(z, theta))[0]
    
    # compute full expression
    express = 4*np.pi*(c/H_z)*(ccl.background.comoving_radial_distance(cosmo_ccl, scale_fact_a)**2)*integral_m
    return express

def n_counts(z_min, z_max, log10mass_min, log10mass_max, theta):
    result = integrate.quad(n_counts_full_integral, z_min, z_max, args=(log10mass_min, log10mass_max, theta))
    return result[0]




################### ~~~ using np.trapz ~~~~~~~~~~###################

def n_counts_trapz_m_integral_only(z, log10mass_min, log10mass_max, theta, n_steps_masses = 1000):
    cosmo_ccl = cosmo(theta)
    
    # compute integral in m
    log10masses_steps = np.linspace(log10mass_min, log10mass_max, num = n_steps_masses)
    dN_dlog10Mdz = halo_mass_function(theta, 10**log10masses_steps, z)
    
    integral_m = np.trapz(dN_dlog10Mdz, log10masses_steps)
    return integral_m


def n_counts_trapz_integral(z_min, z_max, log10mass_min, log10mass_max, theta, n_steps_z = 100, n_steps_masses = 1000):
    
    z_steps = np.linspace(z_min, z_max, num = n_steps_z)
    log10masses_steps = np.linspace(log10mass_min, log10mass_max, num = n_steps_masses)
    N_counts_dz = np.zeros(n_steps_z)
    
    cosmo_ccl = cosmo(theta)
    H0 = 100*theta[3]
    c = ccl.physical_constants.CLIGHT*1e-3 # in km/s

    scale_fact_a = 1/(1+z_steps)
    H_z = H0*ccl.background.h_over_h0(cosmo_ccl, scale_fact_a)
    factor = 4*np.pi*(c/H_z)*(ccl.background.comoving_radial_distance(cosmo_ccl, scale_fact_a)**2)

    for i in range(len(z_steps)):

        dN_dlog10Mdz = halo_mass_function(theta, 10**log10masses_steps, z_steps[i])
        N_counts_dz[i] = factor[i]*np.trapz(dN_dlog10Mdz, log10masses_steps)

    integral_trapz = np.trapz(N_counts_dz, z_steps)    
    
    return integral_trapz



################# ~~~~ FOR SIMULATIONS ~~~~~~ #####################

# generate a random realization of data, given some fiducial
def random_realization_n_counts(N_counts):
    N_counts_random = np.random.poisson(N_counts)
    return N_counts_random
    


#### simulation function ####
# Generate realisation of N counts 
def simulation_seeded(theta, seed, sim_args):
    np.random.seed(seed)
    
    # make them into specified params later
    log10masses = np.linspace(14, 16, num = 5)
    z_min = np.linspace(0.1, 0.9, num = 5)
    z_max = np.linspace(0.2, 1.0, num = 5)
    N_counts_true = np.zeros((len(log10masses)-1, len(z_min)))
    
    N_counts_random = np.zeros((len(log10masses)-1, len(z_min)))
    
    for i in range(len(log10masses) - 1):
        for j in range(len(z_min)):
            N_counts_temp = n_counts_func.n_counts(z_min[j], z_max[j], log10masses[i], log10masses[i+1], theta_fiducial)
            N_counts_true[i][j] = N_counts_temp
            # N_counts_random[i][j] = np.random.poisson(N_counts_temp)
    
    N_counts_random = np.random.poisson(N_counts_true)
    return N_counts_random
    

def simulation_seeded_example(theta, seed, sim_args):
    
    # Pull out data
    auxiliary_data = sim_args[0]
    L = sim_args[1]
    
    # Signal
    mb = apparent_magnitude(theta, auxiliary_data)
        
    # Noise
    np.random.seed(seed)
    noise = np.dot(L, np.random.normal(0, 1, len(L)))
    
    # Return signal + noise
    return mb
















# integrate halo mass function along z
def halo_mass_function_z_bin(theta, mass, z_min, z_max):
    result = integrate.quad(integrand_hmf_z, z_min, z_max, args=(theta, mass))
    
    return result[0]

def integrand_hmf_z(x, theta, mass):
    cosmo_ccl = cosmo(theta)
    scale_fact_a = 1/(1+x)
    H0 = 100*theta[3]
    c = ccl.physical_constants.CLIGHT*1e-3
    H_z = H0*ccl.background.h_over_h0(cosmo_ccl, scale_fact_a)
    
    # using Sheth et al. (1999) halo mass function - but we could extend it to other halo mass functions
    mass_funct = ccl.halos.hmfunc.MassFuncBocquet16(cosmo_ccl)
        
    # calculate halo mass function for a given redshift and a given range of masses
    dn_dlog10M = mass_funct.get_mass_function(cosmo_ccl, mass, scale_fact_a)
        
    express = 4*np.pi*(c/H_z)*(ccl.background.comoving_radial_distance(cosmo_ccl, scale_fact_a)**2)*dn_dlog10M/(mass*np.log(10))
    
    return express


