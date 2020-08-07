import numpy as np
import scipy.integrate as integrate
# numpy trapz
import pyccl as ccl
import time as time
#import simulators.jla_supernovae.jla_parser as jla_parser
from scipy.interpolate import RectBivariateSpline

class Model():

    def __init__(self, log10masses = np.linspace(14, 15.5, num = 4), z_min = np.linspace(0.1, 0.9, num = 5), 
                 z_max = np.linspace(0.2, 1.0, num = 5), data_path = 'N_counts_random_data.npy'):

        
        ## import data
        self.data = np.load(data_path)
        
        #  Omega_{cdm}h^{2}, Omega_{b}h^{2}, sigma_{8}
        self.theta_fiducial = np.array([0.1197, 0.76])
        self.h = 0.69
        self.npar = len(self.theta_fiducial)
        
        # N data points 
        self.ndata = len(self.data)
        
         # make them into specified params later
        self.log10masses = log10masses
        self.log10masses_2d = np.stack((log10masses[:-1], log10masses[1:]))
        self.z_min = z_min
        self.z_max = z_max
        
        # Compute expected values
        
        # FOR NOW
        self.N_counts = self.all_n_counts_trapz(self.z_min, self.z_max, self.log10masses_2d, self.theta_fiducial)
        #self.N_counts = np.array([[1.78484613e+04, 5.76969893e+04, 7.91958165e+04, 7.60278893e+04, 5.88511940e+04],
         #                       [2.53148842e+03, 6.50855639e+03, 6.79903632e+03, 4.77688556e+03, 2.60820123e+03],
          #                      [1.41327071e+02, 2.40973371e+02, 1.54407289e+02, 6.19683804e+01, 1.80223466e+01],
           #                     [1.15793270e+00, 8.97133179e-01, 2.22000351e-01, 2.95440371e-02, 2.45071739e-03]])
        #shape_new = self.N_counts.shape[0]*self.N_counts.shape[1]
        #self.N_counts = self.N_counts.reshape(shape_new) 
        
        
    # Cosmology modulus
    def cosmo(self, theta, h=0.69):
        # create a flat cosmology with massive neutrinos and some other default params
        
        # $\Omega_{m}h^{2} = \Omega_{cdm}h^{2} + \Omega_{b}h^{2}$, $\sigma_{8}$, $h$, $n_{s}$, $w_{0}$}
        Omega_cdm = theta[0]/(h**2)
        Omega_b = 0.02222/(h**2)
        sigma8 = theta[1]
        
        
        # setting up different cosmology configuration to match benchmarks and keep CLASS from crashing.
        cosmo_ccl = ccl.Cosmology(Omega_c= Omega_cdm, Omega_b= Omega_b, sigma8= sigma8,
                                      h= h, n_s= 0.9655, w0= -1,
                                      Omega_g=0, Omega_k=0,
                                      Neff=0, m_nu=0.0,
                                      wa=0, T_CMB=2.7, transfer_function='bbks',mass_function='tinker')
        
        return cosmo_ccl
    
    
    def halo_mass_function(self, cosmo_ccl, masses, z):
        scale_fact_a = 1 / (1 + z)
        
        # using Sheth et al. (1999) halo mass function - but we could extend it to other halo mass functions
        mass_funct = ccl.halos.hmfunc.MassFuncTinker08(cosmo_ccl)
        
        dn_dM = np.array([mass_funct.get_mass_function(cosmo_ccl, masses, a) for a in scale_fact_a])
        return dn_dM
    
    
    ############# ~~~~~~~~~~~~ USING NP.TRAPZ ~~~~~~~~~~~~ ############
    def n_counts_trapz_m_integral_only(self, z, log10mass_min, log10mass_max, theta, n_steps_masses = 1000):
        cosmo_ccl = self.cosmo(theta)

        # compute integral in m
        log10masses_steps = np.linspace(log10mass_min, log10mass_max, num = n_steps_masses)
        dN_dlog10Mdz = self.halo_mass_function(theta, 10**log10masses_steps, z)

        integral_m = np.trapz(dN_dlog10Mdz, log10masses_steps)
        return integral_m
    
    def n_counts_trapz_integral(self, cosmo_ccl, z_steps, mass_grid, theta):
        H0 = 100 * self.h
        c = ccl.physical_constants.CLIGHT * 1e-3  # in km/s
        
        scale_fact_a = 1 / (1 + z_steps)
        H_z = H0 * ccl.background.h_over_h0(cosmo_ccl, scale_fact_a)
        factor = 4 * np.pi * (c / H_z) * (ccl.background.comoving_radial_distance(cosmo_ccl, scale_fact_a) ** 2)
        
        dN_dlog10Mdz = self.halo_mass_function(cosmo_ccl, mass_grid.flatten(), z_steps)
        dN_dlog10Mdz = dN_dlog10Mdz.T.reshape((mass_grid.shape[0], mass_grid.shape[1], z_steps.shape[0]), order='C')
        N_counts_dz = factor[np.newaxis, np.newaxis, :] * np.trapz(dN_dlog10Mdz, np.log10(mass_grid[:, :, np.newaxis]), axis=1)
        
        integral_trapz = np.trapz(N_counts_dz, z_steps, axis=-1)
        return integral_trapz
    
    def all_n_counts_trapz(self, z_min, z_max, log10masses, theta, n_steps_z=100, n_steps_masses=1000):
        N_counts_true = np.zeros((log10masses.shape[1], len(z_min)))
        
        log10mass_mins = log10masses[0]
        log10mass_maxs = log10masses[1]
        
        nm_bins = log10mass_maxs.shape[0]
        delta_bin = log10mass_maxs - log10mass_mins
        mass_grid = 10 ** (log10mass_mins[:, np.newaxis] +
                           np.linspace(0, 1, n_steps_masses) * delta_bin[:, np.newaxis])
        
        cosmo_ccl = self.cosmo(theta)
        t1 = time.process_time()
        for i in range(len(z_min)):
            z_steps = np.linspace(z_min[i], z_max[i], num=n_steps_z)

            N_counts_temp = self.n_counts_trapz_integral(cosmo_ccl, z_steps, mass_grid, theta)
            N_counts_true[:, i] = N_counts_temp
            
        t2 = time.process_time()
        #print(str(t2-t1) + ' s')

        shape_new = N_counts_true.shape[0] * N_counts_true.shape[1]
        N_counts_true = N_counts_true.reshape(shape_new)
        return N_counts_true, cosmo_ccl

    ############# ~~~~~~~~~~~~ FOR SIMULATIONS ~~~~~~~~~~~~ #############
    # Generate realisation of N counts
    def simulation(self, theta, seed, pr = False):
        
        # Set the seed
        np.random.seed(seed)

        t1 = time.process_time()
        N_counts_true, cosmo_ccl = self.all_n_counts_trapz(self.z_min, self.z_max, self.log10masses_2d, theta)
        N_counts_true = np.random.poisson(N_counts_true)
        #N_counts_true = np.round(N_counts_true).astype(int)
        t2 = time.process_time()
        if pr == True:
            print('time to compute true counts: '+ str(t2-t1))
        
        t1 = time.process_time()
        results = [self.get_Mz_array(cosmo_ccl, seed, N_counts_true[i], self.z_min[i], self.z_max[i], self.log10masses[0], self.log10masses[-1]) 
                   for i in np.arange(len(N_counts_true))]
        t2 = time.process_time()
        if pr == True:
            print('time to create a catalogue: '+ str(t2-t1))
        
        t1 = time.process_time()
        results_bins = [self.get_counts(results[i][:, 0]) for i in np.arange(len(results))]
        results_bins = np.concatenate(results_bins).reshape((5, 3)).T.flatten()
        t2 = time.process_time()
        if pr == True:
            print('time to bin a catalogue: '+ str(t2-t1))
        
        return results_bins
    
    ############# ~~~~~~~~~~~~ FOR MOCK (M, z) PAIRS ~~~~~~~~~~~~ #############
    def get_Mz_array(self, cosmo_ccl, seed, N, z_min, z_max, M_min, M_max, n_grid_z= 20, n_grid_z_fine = 2000, 
                     n_grid_M= 1000, n_grid_M_fine = 10000):
        
        np.random.seed(seed) 
        
        z_array = np.linspace(z_min, z_max, num = n_grid_z)
        masses = np.logspace(M_min, M_max, num =  n_grid_M)
        
        mass_funct = ccl.halos.hmfunc.MassFuncTinker08(cosmo_ccl)
        hmf = self.halo_mass_function2(mass_funct, cosmo_ccl, masses, z_array)
        
        z_array_fine = np.linspace(z_min, z_max, num = n_grid_z_fine)
        masses_fine = np.logspace(M_min, M_max, num = n_grid_M_fine)
    
        interp_spline = RectBivariateSpline(z_array, masses, hmf)
        hmf_fine = interp_spline(z_array_fine, masses_fine)
        hmf_sums = hmf_fine.sum(axis = 1, keepdims= True)
        prob_norm_M_z = hmf_fine/hmf_sums
        
        prob_norm_Mz = prob_norm_M_z/np.sum(prob_norm_M_z)
        prob_norm_Mz_flat = prob_norm_Mz.flatten()
        
        indices_array = np.arange(len(prob_norm_Mz_flat))
        indices_values = np.random.choice(indices_array, size = N, p = prob_norm_Mz_flat, replace = False)
    
        m_indices_values = indices_values%n_grid_M_fine
        z_indices_values = indices_values//n_grid_M_fine
        
        masses_vals = np.take(masses_fine, m_indices_values)
        z_vals = np.take(z_array_fine, z_indices_values)
        
        return np.stack((masses_vals, z_vals)).T
    
    
    def halo_mass_function2(self, mass_funct, cosmo_ccl, masses, z):
        scale_fact_a = 1 / (1 + z)
        
        # using Sheth et al. (1999) halo mass function - but we could extend it to other halo mass functions
        # mass_funct = ccl.halos.hmfunc.MassFuncTinker08(cosmo_ccl)
        
        dn_dM = np.array([mass_funct.get_mass_function(cosmo_ccl, masses, a) for a in scale_fact_a])
        return dn_dM
    
    def get_counts(self, arr_m):
        count_1 = len(arr_m[(arr_m >= 10**(14.0)) & (arr_m <10**(14.5))])
        count_2 = len(arr_m[(arr_m >= 10**(14.5)) & (arr_m <10**(15.0))])
        count_3 = len(arr_m[(arr_m >= 10**(15.0))])
        return np.array([count_1, count_2, count_3])
        
        
   




