import numpy as np
import scipy.integrate as integrate
# numpy trapz
import pyccl as ccl
import time as time
#import simulators.jla_supernovae.jla_parser as jla_parser

class Model():

    def __init__(self, log10masses = np.linspace(14, 15.5, num = 4), z_min = np.array([0., 0.25, 0.5, 0.75, 1.]), 
                 z_max = np.array([0.25, 0.5, 0.75, 1., 1.5]), Y_bins_limits = [np.logspace(np.log10(8.6e-12), np.log10(3.9e-9), num = 16),
                 np.logspace(np.log10(4.3e-12), np.log10(5.1e-10), num = 15), np.logspace(np.log10(3.1e-12), np.log10(1.8e-10), num = 16),
                 np.logspace(np.log10(3.1e-12), np.log10(1.1e-10), num = 14), np.logspace(np.log10(2.5e-12), np.log10(6.6e-11), num = 13)],
                 theta_fiducial_mass_calibration = np.array([1.8, 0., 0., 2.4e-10, 1e14, 0., 0., 0.127]),
                 theta_fiducial = np.array([0.1197, 0.76])):

        
        
        #  Omega_{cdm}h^{2}, sigma_{8}
        self.theta_fiducial = theta_fiducial
        self.theta_fiducial_mass_calibration = theta_fiducial_mass_calibration
        self.h = 0.69
        self.npar = len(self.theta_fiducial)+len(self.theta_fiducial_mass_calibration)
        
        
        # make them into specified params later
        self.Y_bins_limits = Y_bins_limits
        self.z_min = z_min
        self.z_max = z_max
        
        # Compute expected values
        
        # FOR NOW
        self.N_counts = self.all_n_counts_trapz(self.z_min, self.z_max, self.Y_bins_limits, self.theta_fiducial, 
                                                self.theta_fiducial_mass_calibration)
        
        
        
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
        hm_def = ccl.halos.MassDef(500, 'critical')
        
        mass_funct = ccl.halos.MassFuncTinker08(cosmo_ccl, mass_def=hm_def)
        dn_dM = np.array([mass_funct.get_mass_function(cosmo_ccl, masses, a) for a in scale_fact_a])
        
        return dn_dM
    
    ############# ~~~~~~~~~~~~ Y_500 functions ~~~~~~~~~~~ ############
    def compute_mass_calibration_params_array(self, cosmo_ccl, theta_mass_calibration, M_500, z):
        
        alpha_Y = theta_mass_calibration[0]
        beta_Y = theta_mass_calibration[1]
        gamma_Y = theta_mass_calibration[2]

        Y_star = theta_mass_calibration[3]
        M_star = theta_mass_calibration[4]

        alpha_sigma = theta_mass_calibration[5]
        gamma_sigma = theta_mass_calibration[6]
        sigma_logY0 = theta_mass_calibration[7]

        scale_fact_a = 1/(1+z)
        

        E_z = (ccl.background.h_over_h0(cosmo_ccl, scale_fact_a)**(2/3))
        D_A_z = (ccl.background.comoving_radial_distance(cosmo_ccl, scale_fact_a)/100)**(-2)

        mean_Y_500 = Y_star*((M_500/M_star)**alpha_Y)*np.exp(beta_Y*(np.log(M_500/M_star)**2))*((1+z)**gamma_Y)*E_z*D_A_z

        sigma_logY_500 = sigma_logY0*((M_500/M_star)**alpha_sigma)*((1+z)**gamma_sigma)

        
        return mean_Y_500, sigma_logY_500

    def compute_probability_Y_500_array(self, mean_Y_500, sigma_logY_500, Y_500_true): 
        prob_Y_500_true = np.exp(-((np.log(Y_500_true) - np.log(mean_Y_500))**2)/(2*(sigma_logY_500**2)))/(np.sqrt(2*np.pi)*sigma_logY_500)/Y_500_true

        return prob_Y_500_true

    def compute_Y500_integral(self, index, Y_bins_limits, mean_Y_500_flat2, sigma_logY_500_flat2, n_steps_M_500, n_steps_z, n_steps_Y):
        n_steps_Mz = n_steps_M_500*n_steps_z
        
        Y_bins_min = Y_bins_limits[index]
        Y_bins_max = Y_bins_limits[index+1]
        Y_bins_temp = np.linspace(Y_bins_min, Y_bins_max, num = n_steps_Y).reshape(n_steps_Y, 1)

        Y_500_array_flat = np.repeat(Y_bins_temp, n_steps_Mz, axis = 1).flatten()

        prob_Y_500_array = self.compute_probability_Y_500_array(mean_Y_500_flat2, sigma_logY_500_flat2, Y_500_array_flat).reshape(n_steps_Y, n_steps_Mz)

        Y500_integral = np.trapz(prob_Y_500_array, Y_bins_temp, axis = 0).reshape(n_steps_M_500, n_steps_z)

        return Y500_integral
    def compute_M500_integral(self, Y500_integral, hmf, log10_M_500_steps):
        Y500_integral = np.array(hmf).T*Y500_integral
        M_500_integral = np.trapz(Y500_integral, log10_M_500_steps, axis = 0)

        return M_500_integral
    def compute_N_integral(self, M_500_integral, factor, z_array):
        N_counts = np.trapz(M_500_integral*factor, z_array.flatten())
    
        return N_counts
    ############# ~~~~~~~~~~~~ USING NP.TRAPZ ~~~~~~~~~~~~ ############
    def all_n_counts_trapz(self, z_min, z_max, Y_bins_limits, theta, theta_mass_calibration, 
                           n_steps_z=100, n_steps_M_500 = 100, n_steps_Y = 100):
        
        n_bins_z = len(z_min)
        n_steps_Mz = n_steps_z*n_steps_M_500 
        
        #log10_M_500_steps = np.linspace(6, 16, num = n_steps_M_500)
        log10_M_500_steps = np.linspace(14, 15.5, num = n_steps_M_500)
        M_500_steps = 10**log10_M_500_steps
        
        M_array = M_500_steps.reshape(n_steps_M_500, 1)
        M_array_flat = np.repeat(M_array, n_steps_z, axis=1).flatten()

        H0 = 100 * 0.69
        c = ccl.physical_constants.CLIGHT * 1e-3  # in km/s
        
        hm_def = ccl.halos.MassDef(500, 'critical')
        cosmo_ccl = self.cosmo(theta)
        nM = ccl.halos.MassFuncTinker08(cosmo_ccl, mass_def=hm_def)

        N_counts = []
        for i in range(n_bins_z):
            z_array = np.linspace(z_min[i], z_max[i], num = n_steps_z).reshape(1, n_steps_z)
            z_array_flat = np.repeat(z_array, n_steps_M_500, axis=0).flatten()

            a = 1/(1+z_array.flatten())
            hmf = [nM.get_mass_function(cosmo_ccl, M_500_steps, a_temp) for a_temp in a]
            H_z = H0 * ccl.background.h_over_h0(cosmo_ccl, a)
            factor = 4 * np.pi * (c / H_z) * (ccl.background.comoving_radial_distance(cosmo_ccl, a) ** 2)

            mean_Y_500_flat, sigma_logY_500_flat = self.compute_mass_calibration_params_array(cosmo_ccl, theta_mass_calibration, M_array_flat, z_array_flat)

            mean_Y_500_flat2 = np.repeat(mean_Y_500_flat.reshape(1, n_steps_Mz), n_steps_Y, axis = 0).flatten()
            sigma_logY_500_flat2 = np.repeat(sigma_logY_500_flat.reshape(1, n_steps_Mz), n_steps_Y, axis = 0).flatten()

            indices = np.linspace(0, len(Y_bins_limits[i])-2, num = len(Y_bins_limits[i])-1).astype(int)
            Y500_integral = [self.compute_Y500_integral(j, Y_bins_limits[i], mean_Y_500_flat2, sigma_logY_500_flat2, n_steps_M_500, n_steps_z, n_steps_Y)
                             for j in indices]

            M500_integral = [self.compute_M500_integral(Y500_integral_temp, hmf, log10_M_500_steps) for Y500_integral_temp in Y500_integral]

            N_counts_temp = np.array([self.compute_N_integral(M_500_integral_temp, factor, z_array) for M_500_integral_temp in M500_integral])
            N_counts.append(N_counts_temp)

            #print(i)
            
        return N_counts


    

    ############# ~~~~~~~~~~~~ FOR SIMULATIONS ~~~~~~~~~~~~ #############
    # Generate realisation of N counts
    def simulation(self, theta, seed):
        
        # Set the seed
        np.random.seed(seed)

        theta_cosmo = theta[:len(self.theta_fiducial)]
        theta_mass_calibration = theta[len(self.theta_fiducial):]
        N_counts_true = self.all_n_counts_trapz(self.z_min, self.z_max, self.Y_bins_limits, theta_cosmo, theta_mass_calibration)
        N_counts_random = [np.random.poisson(N_counts_temp) for N_counts_temp in N_counts_true]
    
        # for debugging purposes
        # N_counts_random = np.random.random(20)
        
        return np.array(N_counts_random)
    



