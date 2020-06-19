import numpy as np
import scipy.integrate as integrate
# numpy trapz
import pyccl as ccl
import time as time
#import simulators.jla_supernovae.jla_parser as jla_parser

class Model():

    def __init__(self, data_path = 'N_counts_random_data.npy'):

        
        ## import data
        self.data = np.load(data_path)
        
        #  Omega_{cdm}h^{2}, Omega_{b}h^{2}, sigma_{8}, h, n_{s}, w_{0}
        self.npar = 6
        self.theta_fiducial = np.array([0.1197, 0.02222, 0.76, 0.69, 0.9655, -1])
        
        # N data points 
        self.ndata = len(self.data)
        
         # make them into specified params later
        self.log10masses = np.linspace(14, 16, num = 5)
        self.z_min = np.linspace(0.1, 0.9, num = 5)
        self.z_max = np.linspace(0.2, 1.0, num = 5)
        
        # Compute expected values
        
        # FOR NOW
        # self.N_counts = self.all_n_counts(self.z_min, self.z_max, self.log10masses, self.theta_fiducial)
        self.N_counts = np.array([[1.78484613e+04, 5.76969893e+04, 7.91958165e+04, 7.60278893e+04, 5.88511940e+04],
                                [2.53148842e+03, 6.50855639e+03, 6.79903632e+03, 4.77688556e+03, 2.60820123e+03],
                                [1.41327071e+02, 2.40973371e+02, 1.54407289e+02, 6.19683804e+01, 1.80223466e+01],
                                [1.15793270e+00, 8.97133179e-01, 2.22000351e-01, 2.95440371e-02, 2.45071739e-03]])
        shape_new = self.N_counts.shape[0]*self.N_counts.shape[1]
        self.N_counts = self.N_counts.reshape(shape_new) 
        
        
    # Cosmology modulus
    def cosmo(self, theta):
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
    
    def halo_mass_function(self, theta, masses, z):
        cosmo_ccl = self.cosmo(theta)
        scale_fact_a = 1/(1+z)
        
        # using Sheth et al. (1999) halo mass function - but we could extend it to other halo mass functions
        mass_funct = ccl.halos.hmfunc.MassFuncSheth99(cosmo_ccl)
        
        # calculate halo mass function for a given redshift and a given range of masses
        dn_dM = mass_funct.get_mass_function(cosmo_ccl, masses, scale_fact_a)
        
        return dn_dM
    
    # integrate halo mass function 
    def n_counts_m_integral(self, log10mass, z, theta):
        cosmo_ccl = self.cosmo(theta)
        scale_fact_a = 1/(1+z)

        # using Sheth et al. (1999) halo mass function - but we could extend it to other halo mass functions
        mass_funct = ccl.halos.hmfunc.MassFuncSheth99(cosmo_ccl)

        # calculate halo mass function for a given redshift and a given range of masses
        dn_dlog10M = mass_funct.get_mass_function(cosmo_ccl, 10**log10mass, scale_fact_a)

        return dn_dlog10M
    
    # integrate halo mass function 
    def n_counts_full_integral(self, z, log10mass_min, log10mass_max, theta):
        cosmo_ccl = self.cosmo(theta)
        scale_fact_a = 1/(1+z)

        H0 = 100*theta[3]
        c = ccl.physical_constants.CLIGHT*1e-3 # in km/s
        H_z = H0*ccl.background.h_over_h0(cosmo_ccl, scale_fact_a)

        # compute integral in m
        integral_m = integrate.quad(self.n_counts_m_integral, log10mass_min, log10mass_max, args=(z, theta))[0]

        # compute full expression
        express = 4*np.pi*(c/H_z)*(ccl.background.comoving_radial_distance(cosmo_ccl, scale_fact_a)**2)*integral_m
        return express

    # return # of counts in a redshift-mass bin
    def n_counts(self, z_min, z_max, log10mass_min, log10mass_max, theta):
        result = integrate.quad(self.n_counts_full_integral, z_min, z_max, args=(log10mass_min, log10mass_max, theta))
        return result[0]
    
    # return an array of all counts for all bins
    def all_n_counts(self, z_min, z_max, log10masses, theta):
        N_counts_true = np.zeros((len(log10masses)-1, len(z_min)))
        for i in range(len(log10masses) - 1):
            for j in range(len(z_min)):
                N_counts_temp = self.n_counts(z_min[j], z_max[j], log10masses[i], log10masses[i+1], theta)
                N_counts_true[i][j] = N_counts_temp
                print(i, j)
        
        shape_new = N_counts_true.shape[0]*N_counts_true.shape[1]
        N_counts_true = N_counts_true.reshape(shape_new)        
        return N_counts_true
    
    
    
    ############# ~~~~~~~~~~~~ USING NP.TRAPZ ~~~~~~~~~~~~ ############
    def n_counts_trapz_m_integral_only(self, z, log10mass_min, log10mass_max, theta, n_steps_masses = 1000):
        cosmo_ccl = self.cosmo(theta)

        # compute integral in m
        log10masses_steps = np.linspace(log10mass_min, log10mass_max, num = n_steps_masses)
        dN_dlog10Mdz = self.halo_mass_function(theta, 10**log10masses_steps, z)

        integral_m = np.trapz(dN_dlog10Mdz, log10masses_steps)
        return integral_m


    def n_counts_trapz_integral(self, z_min, z_max, log10mass_min, log10mass_max, theta, n_steps_z = 100, n_steps_masses = 1000):

        z_steps = np.linspace(z_min, z_max, num = n_steps_z)
        log10masses_steps = np.linspace(log10mass_min, log10mass_max, num = n_steps_masses)
        N_counts_dz = np.zeros(n_steps_z)

        cosmo_ccl = self.cosmo(theta)
        H0 = 100*theta[3]
        c = ccl.physical_constants.CLIGHT*1e-3 # in km/s

        scale_fact_a = 1/(1+z_steps)
        H_z = H0*ccl.background.h_over_h0(cosmo_ccl, scale_fact_a)
        factor = 4*np.pi*(c/H_z)*(ccl.background.comoving_radial_distance(cosmo_ccl, scale_fact_a)**2)

        for i in range(len(z_steps)):

            dN_dlog10Mdz = self.halo_mass_function(theta, 10**log10masses_steps, z_steps[i])
            N_counts_dz[i] = factor[i]*np.trapz(dN_dlog10Mdz, log10masses_steps)

        # N_counts_dz[i] = factor[i]*np.trapz(dN_dlog10Mdz, log10masses_steps) - take it out of loop
        integral_trapz = np.trapz(N_counts_dz, z_steps)    

        return integral_trapz

    def all_n_counts_trapz(self, z_min, z_max, log10masses, theta):
        N_counts_true = np.zeros((len(log10masses)-1, len(z_min)))
        
        t1 = time.process_time()
        for i in range(len(log10masses) - 1):
            for j in range(len(z_min)):
                N_counts_temp = self.n_counts_trapz_integral(z_min[j], z_max[j], log10masses[i], log10masses[i+1], theta)
                N_counts_true[i][j] = N_counts_temp
                
                # print(i, j)
            print(i)
        t2 = time.process_time()
        print(str(t2-t1) + ' s')
        
        shape_new = N_counts_true.shape[0]*N_counts_true.shape[1]
        N_counts_true = N_counts_true.reshape(shape_new)        
        return N_counts_true
    

    ############# ~~~~~~~~~~~~ FOR SIMULATIONS ~~~~~~~~~~~~ #############
    # Generate realisation of N counts
    def simulation(self, theta, seed):
        
        # Set the seed
        np.random.seed(seed)

        
        N_counts_true = self.all_n_counts_trapz(self.z_min, self.z_max, self.log10masses, theta)
        N_counts_random = np.random.poisson(N_counts_true)
        
    
        # for debugging purposes
        # N_counts_random = np.random.random(20)
        
        return N_counts_random
    



