import numpy as np
import scipy.integrate as integrate
# numpy trapz
import pyccl as ccl
import time as time
#import simulators.jla_supernovae.jla_parser as jla_parser

class Model():

    def __init__(self, log10masses = np.linspace(14, 15.5, num = 4), z_min = np.linspace(0.1, 0.9, num = 5), 
                 z_max = np.linspace(0.2, 1.0, num = 5), data_path = 'N_counts_random_data.npy'):

        
        ## import data
        self.data = np.load(data_path)
        
        #  Omega_{cdm}h^{2}, Omega_{b}h^{2}, sigma_{8}, h, n_{s}, w_{0}
        self.npar = 2
        self.theta_fiducial = np.array([0.1197, 0.76]) #, 0.69, 0.9655, -1])
        self.h = 0.69
        
        # N data points 
        self.ndata = len(self.data)
        
         # make them into specified params later
        self.log10masses = log10masses
        self.z_min = z_min
        self.z_max = z_max
        
        # Compute expected values
        self.cosmo_object = self.cosmo(self.theta_fiducial)
        self.hmf_object = self.halo_mass_function_object(self.theta_fiducial)
        
        # FOR NOW
        self.N_counts = self.n_counts_trapz_all(self.theta_fiducial, self.z_min, self.z_max, self.log10masses)
        #self.N_counts = np.array([[1.78484613e+04, 5.76969893e+04, 7.91958165e+04, 7.60278893e+04, 5.88511940e+04],
        #[2.53148842e+03, 6.50855639e+03, 6.79903632e+03, 4.77688556e+03, 2.60820123e+03],
         #       [1.41327071e+02, 2.40973371e+02, 1.54407289e+02, 6.19683804e+01, 1.80223466e+01],
          #                      [1.15793270e+00, 8.97133179e-01, 2.22000351e-01, 2.95440371e-02, 2.45071739e-03]])
        #shape_new = self.N_counts.shape[0]*self.N_counts.shape[1]
        #self.N_counts = self.N_counts.reshape(shape_new) 
        
        
    # Cosmology modulus
    def cosmo(self, theta, h=0.69):
        # create a flat cosmology with massive neutrinos and some other default params
        # h = 0.69
        # $\Omega_{m}h^{2} = \Omega_{cdm}h^{2} + \Omega_{b}h^{2}$, $\sigma_{8}$, $h$, $n_{s}$, $w_{0}$}
        Omega_cdm = theta[0]/(h**2)
        Omega_b = 0.02222/(h**2)
        sigma8 = theta[1]
        
        
        #n_s = theta[4]
        #w0 = theta[5]
        
        # setting up different cosmology configuration to match benchmarks and keep CLASS from crashing.
        cosmo_ccl = ccl.Cosmology(Omega_c= Omega_cdm, Omega_b= Omega_b, sigma8= sigma8,
                                      h= h, n_s= 0.9655, w0= -1,
                                      Omega_g=0, Omega_k=0,
                                      Neff=0, m_nu=0.0,
                                      wa=0, T_CMB=2.7, transfer_function='bbks',mass_function='tinker')
        
        return cosmo_ccl
    
    def halo_mass_function_object(self, theta):
        cosmo_ccl = self.cosmo(theta)
        
        # using Sheth et al. (1999) halo mass function - but we could extend it to other halo mass functions
        mass_funct = ccl.halos.hmfunc.MassFuncSheth99(cosmo_ccl)
        
        return mass_funct
    
    def halo_mass_function(self, cosmo_ccl, mass_funct, masses, z):
        scale_fact_a = 1/(1+z)
        
        # using Sheth et al. (1999) halo mass function - but we could extend it to other halo mass functions
        # mass_funct = ccl.halos.hmfunc.MassFuncSheth99(cosmo_ccl)
        
        # calculate halo mass function for a given redshift and a given range of masses
        dn_dM = mass_funct.get_mass_function(cosmo_ccl, masses, scale_fact_a)
        
        return dn_dM
    
    
    ############### ~~~~~~~~~~ START NEW FUNCTIONS ~~~~~~~~~~ ####################
    def n_counts_trapz(self, theta, z_min, z_max, log10mass_min, log10mass_max, n_steps_z = 100, n_steps_masses = 1000):
        tic = time.process_time()
        cosmo_ccl = self.cosmo(theta)
        toc = time.process_time()
        print('computing cosmo: ' + str((toc-tic)) + ' s')
        
        tic = time.process_time()
        hmf_object_temp = ccl.halos.hmfunc.MassFuncSheth99(cosmo_ccl)
        toc = time.process_time()
        print('computing halo mass function object: ' + str((toc-tic)) + ' s')
        
        tic = time.process_time()
        N_counts_dz = np.zeros(n_steps_z)
        log10masses_steps = np.linspace(log10mass_min, log10mass_max, num = n_steps_masses)
        z_steps = np.linspace(z_min, z_max, num = n_steps_z)
        
        H0 = 100*self.h
        c = ccl.physical_constants.CLIGHT*1e-3 # in km/s

        scale_fact_a = 1/(1+z_steps)
        H_z = H0*ccl.background.h_over_h0(cosmo_ccl, scale_fact_a)
        factor = 4*np.pi*(c/H_z)*(ccl.background.comoving_radial_distance(cosmo_ccl, scale_fact_a)**2)
        toc = time.process_time()
        print('computing factors: ' + str((toc-tic)) + ' s')

        tic = time.process_time()
        for i in range(len(z_steps)):
            dN_dlog10Mdz = hmf_object_temp.get_mass_function(cosmo_ccl, 10**log10masses_steps, scale_fact_a[i])
            N_counts_dz[i] = factor[i]*np.trapz(dN_dlog10Mdz, log10masses_steps)

        N_counts_trapz = np.trapz(N_counts_dz, z_steps)    
        toc = time.process_time()
        print('loop over z_steps: ' + str((toc-tic)) + ' s')
        return N_counts_trapz
    
    def n_counts_trapz_all(self, theta, z_min, z_max, log10masses, n_steps_z = 100, n_steps_masses = 1000):
        
        t1 = time.process_time()
        N_counts_true = np.zeros((len(log10masses)-1, len(z_min)))
        
        H0 = 100*self.h
        c = ccl.physical_constants.CLIGHT*1e-3 # in km/s
        
        #tic = time.process_time()
        cosmo_ccl = self.cosmo(theta)
        #toc = time.process_time()
        #print('computing cosmo: ' + str((toc-tic)) + ' s')
        
        #tic = time.process_time()
        hmf_object_temp = ccl.halos.hmfunc.MassFuncSheth99(cosmo_ccl)
        #toc = time.process_time()
        #print('computing halo mass function object: ' + str((toc-tic)) + ' s')
        
        #tic = time.process_time()
        for i in range(len(log10masses) - 1):
            log10masses_steps = np.linspace(log10masses[i], log10masses[i+1], num = n_steps_masses)
            
            for j in range(len(z_min)):
                N_counts_dz = np.zeros(n_steps_z)
                z_steps = np.linspace(z_min[j], z_max[j], num = n_steps_z)
                
                scale_fact_a = 1/(1+z_steps)
                H_z = H0*ccl.background.h_over_h0(cosmo_ccl, scale_fact_a)
                factor = 4*np.pi*(c/H_z)*(ccl.background.comoving_radial_distance(cosmo_ccl, scale_fact_a)**2)
                
                for k in range(n_steps_z):
                    dN_dlog10Mdz = hmf_object_temp.get_mass_function(cosmo_ccl, 10**log10masses_steps, scale_fact_a[k])
                    N_counts_dz[k] = factor[k]*np.trapz(dN_dlog10Mdz, log10masses_steps)
                
                #print(i, j)
                N_counts_true[i][j] = np.trapz(N_counts_dz, z_steps)
            #print(i)
        
        #toc = time.process_time()
        #print('loop over z_steps: ' + str((toc-tic)) + ' s')
        
        shape_new = N_counts_true.shape[0]*N_counts_true.shape[1]
        N_counts_true = N_counts_true.reshape(shape_new)
        
        t2 = time.process_time()
        # print('total: ' + str(t2-t1) + ' s')
        return N_counts_true
    
    ############### ~~~~~~~~~~~ END NEW FUNCTIONS ~~~~~~~~~~  ####################
    

    ############# ~~~~~~~~~~~~ FOR SIMULATIONS ~~~~~~~~~~~~ #############
    # Generate realisation of N counts
    def simulation(self, theta, seed):
        
        # Set the seed
        np.random.seed(seed)

        N_counts_true = self.n_counts_trapz_all(theta, self.z_min, self.z_max, self.log10masses)
        N_counts_random = np.random.poisson(N_counts_true)
        
    
        # for debugging purposes
        # N_counts_random = np.random.random(20)
        
        return N_counts_random
    



