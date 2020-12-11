import json
import os
import time

from autograd import numpy as np
from autograd import scipy as sp
from autograd import grad
from autograd.misc.optimizers import adam

import wandb


class HMC:
    """
    Implementation of Hamiltonian-Montecarlo (HMC) sampling
    using Euclidean-Gaussian kinetic energy.
    """

    def __init__(self,
        log_target_func, position_init,
        total_samples=1000, leapfrog_steps=20, step_size=1e-1,
        burn_in=0.1, thinning_factor=1,
        mass=1.0, random_seed=None, progress=False,
        wb_settings=False,
    ):
        """
        Perform HMC using a Euclidean-Gaussian kinetic energy.
        
        log_target_func:

        position_init:

        total_samples, burn_in, thinning_factor:
            Number of desired samples (after burn-in and thinning).
            Burn-in should be a proportion between 0 and 1.
            Thinning factor should be an integer.
        
        leapfrog_steps, step_size, mass:
            Hyperparameters for the Euclidian-Gaussian HMC.

        random_seed:
            (int or None) Random seed for the sampler.

        progress:
            (False or int) How often to print progress.

        wb_settings:
            (False or dict) Settings for logging to Weights & Biases (optional).
            entity: Username of the project host (by default, uses gpestre/am207 shared project).
            project: Name of the project (by default, uses gpestre/am207 shared project).
            group: Name of the group (e.g. hmc_bnn_lv) 
            name: Short name for this particular run.
            notes: Short note describing the tuning for this particular run.
                (Note: The full dictionary of hyperparameters will also be uploaded.)
            save_code: Whether or not to upload a copy of the script/notebook HMC called run from.
                (If True, also requires allowing code uploads in the settings of your W&B account.)
            [The options above are sent to DeepNote's init function -- see here: https://docs.wandb.com/library/init .]
            [The options below are additional parameters for uploading samples and performance metrics to W&B .]
            progress: Integer indicating how often to update progress (e.g. every 1000 steps).
            base_path: Local directory where samples will be saved before W&B upload.
                (If blank, uses the folder wandb creates for this run;
                the default is a good choice, but does not seem to work on DeepNote.)
            filename: Name of the file the HMC samples/state are dumped to (default: "hmc_state.json").
            archive: A static dictionary of params/values to archive (e.g. info about the priors).
                (These are logged as hyperparameters addition to the HMC intialization parameters).
        """

        # Initialize W & B logging (optional):
        self.wb_settings = wb_settings
        self.wb_progress = None if (not self.wb_settings) or ('progress' not in wb_settings) else wb_settings['progress']
        if self.wb_settings is not False:
            # Create a dictionary of hyperparameters:
            config = dict() if 'archive' not in wb_settings else wb_settings['archive']
            config.update({
                'total_samples' : total_samples,
                'leapfrog_steps' : leapfrog_steps,
                'step_size' : step_size,
                'burn_in' : burn_in,
                'thinning_factor' : thinning_factor,
                'mass' : mass,
                'random_seed' : random_seed,
            })
            # Define helper function:
            def get_wb_setting(key, default):
                if key in wb_settings.keys():
                    return wb_settings[key]
                return default
            # Initialize a W&B run:
            wandb.init(
                entity    = get_wb_setting(key='entity', default='gpestre'),
                project   = get_wb_setting(key='project', default='am207'),
                group     = get_wb_setting(key='group', default='hmc'),
                name      = get_wb_setting(key='name', default='hmc'),
                save_code = get_wb_setting(key='save_code', default=False),
                notes     = get_wb_setting(key='notes', default=None),
                config    = config,  # Dictionary of the hyperparameters.
            )
        
        # Check parameters:
        assert thinning_factor==int(thinning_factor), "thinning_factor must be integer."
        assert (burn_in>=0) and (burn_in<1), "Burn in must be between 0 and 1."
        assert len(position_init.shape)==1 or position_init.shape[0]==1
        
        # Represent position as a 1-by-D matrix:
        position_init = position_init.reshape(1,-1)
        dims = position_init.shape[1]
        
        # Calculate number of samples for given burn-in and thinning:
        iters = HMC.calc_iters(total_samples=total_samples, burn_in=burn_in, thinning_factor=thinning_factor)
        self.burn_num = iters - (total_samples*thinning_factor)
        
        # Store parameters:
        self.log_target_func = log_target_func
        self.position_init = position_init
        self.total_samples = total_samples
        self.leapfrog_steps = leapfrog_steps
        self.step_size = step_size
        self.burn_in = burn_in
        self.thinning_factor = thinning_factor
        self.mass = mass
        self.random_seed = random_seed
        self.progress = progress
        self.dims = dims
        self.iters = iters

        # Build placeholder for state variables:
        self.raw_samples = None  # History of samples after burn-in and thinning (built as list of arrays and converted to numpy 2D array).
        self.samples = None  # History of raw samples (built as list of arrays and converted to numpy 2D array).
        self.n_accepted = None  # Number of accepted samples.
        self.n_rejected = None  # Number of rejected samples.
        self.seconds = None  # Runtime in seconds.
        self.np_random = None  # Numpy random state.

        # Initalize:
        self._reset(warm_start=False)

    def _reset(self, warm_start=False):

        if warm_start:

            # Convert state variables from arrays back to lists
            #   to continue appending:
            if not isinstance(self.raw_samples, list):
                self.raw_samples = self.raw_samples.tolist()
                self.samples = self.samples.tolist()

        else:

            # Reset state history:
            self.raw_samples = list()
            self.samples = list()
            self.n_accepted = 0
            self.n_rejected = 0
            self.seconds = float("-Inf")

            # Reset random state:
            self.np_random = np.random.RandomState(self.random_seed)

    @staticmethod
    def calc_iters(total_samples, burn_in=0.0, thinning_factor=1):
        """
        Compute how many iterations are needed to acheive `total_samples`
        after removing burn-in and performinng thinning.
        """
        assert (burn_in>=0) and (burn_in<1), "burn_in must be a proportion (less than 1)."
        assert isinstance(thinning_factor, int) and thinning_factor>0, "thinning_factor must be a positive integer."
        iters = int(np.ceil(total_samples/(1.0-burn_in)*thinning_factor))
        return iters

    def burn_thin(self):
        """
        Remove burn-in (by negative indexing) and perform thinning (by list slicing).
        Takes builds `.samples` from `.raw_samples`.
        """
        # Preserve types:
        if isinstance(self.raw_samples, list):
            self.samples = self.raw_samples[self.burn_num::self.thinning_factor]
        else:
            # 3D numpy array:
            raw_samples = self.raw_samples.tolist()
            samples = raw_samples[self.burn_num::self.thinning_factor]
            self.samples = np.array(samples)
        
        
    def kinetic_func(self, p):
        """
        Calculate the Euclidean-Guassian kinetic energy of
        a particle of mass `m` with momentum `p`
        (given as a D-dimensional numpy.array).
        """
        #####
        ## Using the fact that M is diagonal by construction:
        ##   M_determinant := (1/m)**D
        ##   M_inverse := 1/m * Identity
        ##   p_transpose @ M_inverse @ p := 1/m * p_transpose @ p
        #####
        m = self.mass
        D = p.shape[1]
        term_inv = 0.5*( np.sum(p**2)*1/m )
        term_det = 0.5*np.log( (1/m)**D )
        term_scalar = 0.5*D*np.log(2*np.pi)
        K = term_inv + term_det + term_scalar
        return K
    
    def kinetic_grad(self, p):
        """
        Calculate the gradient of the Euclidean-Guassian kinetic energy
        with respect to momentum, for a particle with a given mass.
        """
        m = self.mass
        return 1.0/m * p
    
    def potential_func(self, q):
        """
        Calculate the potential energy function
        as the Gibbs distriubtion of the target PDF.
        """
        # return -np.log(target_func(q))
        result = -self.log_target_func(q)
        try:
            # If result is 1-by-1 result.
            result = result.flatten()
            assert len(result)==1
            result = result[0]
        except:
            pass
        return result
    
    # Define the gradient of the potential energy with respect to position:
    def potential_grad(self, q):
        # target_grad = grad(target_func,argnum=0)
        # return - target_grad(q) / target_func(q)
        # Get the logarithmic gradient:  d/dx ln(f(x)) = 1/f(x) * d/dx f(x)  (by chain rule)
        log_target_grad = grad(self.log_target_func,argnum=0)
        return - log_target_grad(q)
    
    # Define sampling distriubution for momentum:
    def momentum_sample(self):
        return self.np_random.normal(loc=0, scale=self.mass, size=(1, self.dims))

    def sample(self, new_samples=None, progress=None):

        # Optionally override parameters from initialization:
        progress = self.progress if progress is None else progress
        if new_samples is not None:
            if not isinstance(new_samples, int) or new_samples<=0:
                raise ValueError('new_samples should be a positive integer.')
            if len(self.raw_samples)==0:
                raise RuntimeError("The `new_samples` parameter should only be used for additional runs.")
            # Prepare for warm start:
            self._reset(warm_start=True)  # Convert arrays back to lists, for appending.
            # Assume brun-in was performed on initial run:
            iters = HMC.calc_iters(total_samples=new_samples, burn_in=0.0, thinning_factor=self.thinning_factor)
            # Start from last (raw) sample:
            position_init = np.vstack(self.raw_samples[-1:])
        else:
            if len(self.raw_samples)>0:
                raise RuntimeError("Initial run has already been performed; To get extra samples from a warm state, try `hmc.sample(new_samples=n)` .")
            # Initialize:
            self._reset(warm_start=False)  
            new_samples = self.total_samples
            iters = self.iters
            position_init = self.position_init
        
        # Start timer:
        time_start = time.time()
        
        # Iterate for specified number of samples:
        q_curr = position_init
        U_curr = None  # Reuse previous result (or set to None to recalculate).
        for i in range(1,1+iters):
            
            # Sample a random momentum:
            #q_curr = self.raw_samples[-1]
            p_curr = self.momentum_sample()
            
            # Take specified number of steps:
            q_prop = q_curr
            p_prop = p_curr
            #hist_q_prop = []
            #hist_p_prop = []
            for j in range(1,1+self.leapfrog_steps):
                # # Keep history for debugging:
                # hist_q_prop.append(q_prop)
                # hist_p_prop.append(p_prop)
                # Leapfrog intergrator:
                p_prop = p_prop - self.step_size/2 * self.potential_grad(q_prop)
                q_prop = q_prop + self.step_size * self.kinetic_grad(p_prop)
                p_prop = p_prop - self.step_size/2 * self.potential_grad(q_prop)
            
            # Reverse momentum:
            p_prop = - p_prop
            
            # Calculate total energy (current):
            if U_curr is None:
                U_curr = self.potential_func(q_curr)
            K_curr = self.kinetic_func(p_curr)
            H_curr = U_curr + K_curr
            
            # Calculate total energy (proposed):
            U_prop = self.potential_func(q_prop)
            K_prop = self.kinetic_func(p_prop)
            H_prop = U_prop + K_prop

            # Calculate acceptance threshold:
            alpha = min(1, np.exp( H_curr - H_prop ))
            
            # Accept or reject:
            u = self.np_random.uniform(0,1)
            if u <= alpha:
                q_curr = q_prop
                U_curr = None  # Trigger recalculation.
                self.n_accepted += 1
            else:
                self.n_rejected += 1
            self.raw_samples.append(q_curr.copy())

            # For debugging:
            if not ( np.all(np.isfinite(p_prop)) and np.all(np.isfinite(q_prop)) ):
                error_msg = f"ERROR: Encountered nan or inf values (iteration {i:,})."
                error_msg += f"\n    q_curr : {q_curr}"
                error_msg += f"\n    p_curr : {p_curr}"
                error_msg += f"\n    U_curr : {U_curr}"
                error_msg += f"\n    K_curr : {K_curr}"
                error_msg += f"\n    q_prop : {q_prop}"
                error_msg += f"\n    p_prop : {p_prop}"
                error_msg += f"\n    U_prop : {U_prop}"
                error_msg += f"\n    K_prop : {K_prop}"
                print(error_msg)
                # print("    hist_q_prop :",np.array(hist_q_prop))
                # print("    hist_p_prop :",np.array(hist_p_prop))
                if self.wb_settings:
                    wandb.alert(
                        title = "HMC failure",
                        level = wandb.AlertLevel.ERROR,
                        text = error_msg,
                    )
                break
            
            if progress and (i % progress == 0):
                time_now = time.time()
                try:
                    time_progress = time_now-time_last
                except:
                    time_progress = time_now-time_start
                time_last = time_now
                print("  iteration {:,}/{:,} ({:,} sec): total {:,} samples ({:.1f}% acceptance)".format(
                    i,
                    iters,
                    int(time_progress),
                    len(self.raw_samples),
                    100 * (self.n_accepted)/(self.n_accepted+self.n_rejected)
                ))
            if self.wb_progress and (i % self.wb_progress == 0):
                self.burn_thin()  # (Re)build list of clean samples from raw_samples.
                # Upload performance metrics:
                wandb.log({
                    'n_samples' : len(self.samples),
                    'n_accepted' : self.n_accepted,
                    'n_rejected' : self.n_rejected,
                    'acceptance_rate' : self.n_accepted/(self.n_accepted+self.n_rejected),
                    'log_target_func' : self.log_target_func(q_curr),
                    'kinetic_grad_mag' : np.linalg.norm(self.kinetic_grad(p_curr)),
                    'potential_grad_mag' : np.linalg.norm(self.potential_grad(q_curr)),
                    'H_curr' : H_curr,
                    'U_curr' : U_curr,
                    'K_curr' : K_curr,
                }, step=i)
                # Save samples/state and upload them:
                # Note: By default, wb_base_path is a local folder created automatically by W&B for this run
                #       but for some reason this causes issues when running on DeepNote,
                #       so the use can also specify some other local folder in wb_settings.
                base_path = wandb.run.dir if 'base_path' not in self.wb_settings else self.wb_settings['base_path']
                # Save HMC samples/state:
                filename = "hmc_state.json" if 'filename' not in self.wb_settings else self.wb_settings['filename']
                filepath = os.path.join(base_path, filename)
                try:
                    self.save_state(filepath, replace=True)  # Saves a json file locally.
                    wandb.save(filepath, base_path=base_path)  # Uploads the file to W&B.
                except Exception as e:
                    print(f"Failed to save {filepath} at step {i}.\n\t{e}")

        if self.wb_settings:
            try:
                wandb.run.finish()
            except:
                print("W & B run already ended. (Should only happen if .sample() is called more than once.)")
                
        # Remove burn-in (by negative indexing) and perform thinning (by list slicing):
        self.raw_samples = np.vstack(self.raw_samples)
        self.burn_thin()  # (Re)builds self.samples from self.raw_samples.
        
        # Stop timer:
        time_end = time.time()
        seconds = time_end-time_start
        if progress:
            print("Finished in {:,} seconds".format(int(seconds)))
        self.seconds += seconds  # Increment timer.

        # Return new samples:
        return self.samples

    def save_state(self, filepath, replace=False):
        # Get raw samples (as list of list of lists):
        raw_samples = np.array(self.raw_samples).tolist()
        # Get samples (as list of list of lists):
        samples = np.array(self.samples).tolist()
        # Get random state:
        random_state = list(self.np_random.get_state())
        random_state[1] = random_state[1].tolist()
        # Make dictionary:
        hmc_state = {
            'raw_samples' : raw_samples,
            'samples' : samples,
            'random_state' : random_state,
        }
        # Save dictionary as json:
        filename = filepath.split('/')[-1]
        directory = filepath[:-len(filename)]
        if not os.path.isdir('./'+directory):
            raise FileNotFoundError(f"Make sure directory exists: {directory}")
        if not replace and os.path.isfile(filename):
            raise FileExistsError(f"Operation would overwrite file: {filepath}")
        with open(filepath, 'w') as f:
            json.dump(hmc_state, f, indent=4)
        print(f"Saved HMC state : {filepath} .")
        
    def load_state(self, filepath):
        # Load dictionary:
        with open(filepath, 'r') as f:
            hmc_state = json.load(f)
        # Get raw samples:
        raw_samples = np.vstack(hmc_state['raw_samples'])
        self.raw_samples = raw_samples
        # Get samples:
        samples = np.vstack(hmc_state['samples'])
        self.samples = samples
        # Get random state:
        random_state = hmc_state['random_state']
        random_state[1] = np.array(random_state[1])
        random_state = tuple(random_state)
        self.np_random.set_state(random_state)
        print(f"Loaded HMC state : {filepath} .")
    

class BBVI:

    """
    Peform Black-Box Variational Inference with the ADAM optimizer
        using the Local Reparameterization trick described in 
        Kingma et al. (2015): https://arxiv.org/abs/1506.02557
    Implementation based on AM207 exercises, drawing from 
        HIPS's autograd: https://github.com/HIPS/autograd/blob/master/examples/black_box_svi.py
    Uses the a multivariate guassian and the variational (which satisfies the mean-field assumption).
    """

    def __init__(self,
        log_target_func, Mu_init, Sigma_init, mode='BNN',
        num_samples=1_000, step_size=0.1, num_iters=1_000,
        random_seed=None, progress=False,
    ):
        """
        Initialize BBVI.

        log_target_func :
            The function that is being approximated.
        
        Mu_init :
            Initialization value for the mean of the variational distribution.
            Expects Mu as a length-D vector or 1-by-D matrix.
        
        Sigma_init :
            Initialization value for the covariance of the variational distribution.
            Expects Sigma as a legnth-D matrix, 1-by-D matrix, or D-by-D diagonal matrix (off-diagonal entries are assumed to be zero).

        mode :
            The type of neural net (determines which ELBO is used).
            Options are: 'BNN', 'BNNLV'.
        
        num_samples :
            Number of noise samples taken during the reparameterization step.
        
        step_size :
            Step size for graident descent.
        
        num_iters :
            Number of steps of gradient desecent.
        
        random_seed :
            Seed for a fixed numpy random state (or None).
        
        progress :
            Number of iterations at which to print progress updates (or False).
        """

        # Check mode:
        mode = mode.replace('+','').upper()
        assert mode in {'BNN','BNNLV'}

        # Check dimensions of Mu:
        try:
            if len(Mu_init.shape)==1:
                Mu_init = Mu_init.reshape(1,-1)
            elif len(Mu_init.shape)==2 and Mu_init.shape[0]==1:
                pass
            else:
                raise ValueError
        except:
            raise ValueError("Expects Mu as a length-D vector or 1-by-D matrix.")

        # Check dimensions of Sigma:
        try:
            if len(Sigma_init.shape)==1:
                Sigma_init = Sigma_init.reshape(1,-1)  # Convert to 1-by-D array.
            elif len(Sigma_init.shape)==2 and Sigma_init.shape[0]==1:
                pass
            elif len(Sigma_init.shape)==2 and Sigma_init.shape[0]==Sigma_init.shape[1]:
                Sigma_init = np.diag(Sigma_init)
            else:
                raise ValueError
        except:
            raise ValueError("Expects Sigma as a legnth-D matrix, 1-by-D matrix, or D-by-D diagonal matrix (off-diagonal entries are assumed to be zero).")
        
        # Check that Mu and Sigma match:
        assert Mu_init.shape[1]==Sigma_init.shape[1], "Expect Mu and Sigma to have same dimension."
        dims = Mu_init.shape[1]
        # Convert covariance to log standard deviation:
        logStDev_init = 0.5*np.log(Sigma_init)

        # Store parameters:
        self.mode = mode
        self.log_target_func = log_target_func
        self.Mu_init = Mu_init
        self.Sigma_init = Sigma_init
        self.logStDev_init = logStDev_init
        self.num_samples = num_samples
        self.step_size = step_size
        self.num_iters = num_iters
        self.progress = progress
        self.random_seed = random_seed
        self.dims = dims

        # Build placeholder for state variables:
        self.params_hist = None  # History of parameters at each interation (built as list of arrays and converted to numpy 2D array).
        self.gradident_hist = None  # History of gradient at each iteration (built as list of arrays and converted to numpy 2D array).
        self.elbo_hist = None  # History of ELBO value at each iteration (built as list and converted to numpy 1D array).
        self.magnitude_hist = None  # History of graident magnitude at each iteration (built as list and converted to numpy 1D array).
        self.seconds = None  # Runtime in seconds.
        self.np_random = None  # Numpy random state.
        self.variational_gradient = None  # Graident function (computed with autograd).
        
        # Initialize:
        self._reset(warm_start=False)

    def _reset(self, warm_start=False):

        if warm_start:

            # Convert state variables from arrays back to lists
            #   to continue appending:
            if not isinstance(self.params_hist, list):
                self.params_hist = self.params_hist.tolist()
                self.gradident_hist = self.gradident_hist.tolist()
                self.elbo_hist = self.elbo_hist.tolist()
                self.magnitude_hist = self.magnitude_hist.tolist()

        else:

            # Reset state history:
            self.params_hist = list()
            self.gradident_hist = list()
            self.elbo_hist = list()
            self.magnitude_hist = list()
            self.seconds = 0

            # Reset random state:
            self.np_random = np.random.RandomState(self.random_seed)

            # Take gradient of objective:
            self.variational_gradient = grad(self.variational_objective, argnum=0)

    def _stack(self, Mu, logStDev):
        """
        Turns Mu and logStDev into a 1-by-2*D matrix of parameters.
        (This is the interal version of the function that uses flat vectors and log-standard-devations;
        a user-facing version that uses covariance and (optionally) square matrics is provided as a class method.)
        """
        params = np.concatenate([Mu,logStDev], axis=-1)
        return params
        
    def _unstack(self, params):
        """
        Extracts Mu and logStDev from a 1-by-2*D matrix of parameters.
        (This is the interal version of the function that uses flat vectors and log-standard-devations;
        a user-facing version that uses covariance and (optionally) square matrics is provided as a class method.)
        """
        Mu = params[:,:self.dims]
        logStDev = params[:,self.dims:]
        return Mu, logStDev

    # Define optimizer and callback:
    def _callback(self, params, iteration, gradient):
        # Calculate ELBO:
        elbo_value = -self.variational_objective(params, iteration)
        elbo_value = elbo_value[0]  # Unpack single value from matrix.
        # Calculate magnitude of graident:
        grad_mag = np.linalg.norm(self.variational_gradient(params, iteration))
        # Update history:
        self.params_hist.append(params)
        self.gradident_hist.append(gradient)
        self.elbo_hist.append(elbo_value)
        self.magnitude_hist.append(grad_mag)
        # Print progress (optional):
        if self.progress and ((iteration+1) % self.progress == 0):
            
            printout = "Iteration {} : lower bound = {}, graident magnitude = {}".format(iteration+1, elbo_value, grad_mag)
            print(printout)
            #print("params :",params)
            #print("gradient :",gradient)

    def gaussian_entropy(self, logStDev):
        """
        Calculates the Gaussian Entropy of logStDev
        (assuming logStDev is diagonal, and passed as a 1-by-D matrix).
        """
        # See : https://statproofbook.github.io/P/mvn-dent
        # Note that since the covariance matrix is diagonal: log(det(Covar)) is equivalent to trace(logStDev).
        # Here, logStDev is a vector of the logs of the square roots
        # of the diagonal entries of the covariange matrix.
        # (i.e. logStDev is standard deviations, whereas the typical G.E. formula uses the covariance matrix).
        #dims = logStDev.shape[-1]
        #return dims/2*np.log(2*np.pi) + 1/2*np.sum(logStDev**2,axis=-1) + 1/2*dims
        return 0.5*self.dims*( 1.0 + np.log(2*np.pi) ) + np.sum(logStDev,axis=-1)

    def variational_objective(self, params, iteration=None):
        """
        Provides a stochastic estimate of the variational lower bound.
        (The `iteration` parameter is required by ADAM but is not used.)
        """
        if self.mode=='BNN':
            Mu, logStDev = self._unstack(params)
            eps_S = self.np_random.randn(self.num_samples,*logStDev.shape)  # Each row is a different sample.
            StDev = np.exp(logStDev)
            W_S = eps_S * StDev + Mu  # Perturb StDev element-wise for each of `num_samples` in eps_S.
            posterior_term = np.mean(self.log_target_func(W_S), axis=0)
            gaussian_entropy_term = self.gaussian_entropy(logStDev)
            elbo_approx = posterior_term + gaussian_entropy_term
            return -elbo_approx
        elif self.mode=='BNNLV':
            raise NotImplementedError("To do.")
        else:
            raise NotImplementedError("Invalid mode: {}".format(self.mode))

    def run(self, num_iters=None, progress=None, warm_start=False):

        # Optionally override parameters from initialization:
        if num_iters is not None:
            self.num_iters = num_iters
        if progress is not None:
            self.progress = progress

        # Create empty lists of state history (for fresh start)
        # or convert them from arrays to lists (for appending):
        self._reset(warm_start=warm_start)
        
        # Start timer:
        time_start = time.time()
        
        # Represent position as a 1-by-2D matrix:
        params_init = self._stack(self.Mu_init, self.logStDev_init)

        # Perform optimization with ADAM:
        params = adam(
            grad = self.variational_gradient,
            x0 = params_init,
            step_size = self.step_size,
            num_iters = self.num_iters,
            callback = self._callback,
        )
        
        # Stop timer:
        time_end = time.time()
        seconds = time_end-time_start
        if self.progress:
            print("Finished in {:,} seconds".format(int(seconds)))
        self.seconds += seconds  # Increment timer.
        
        # Convert results to numpy arrays:
        self.params_hist = np.vstack(self.params_hist)
        self.gradident_hist = np.vstack(self.gradident_hist)
        self.elbo_hist = np.vstack(self.elbo_hist)
        self.magnitude_hist = np.vstack(self.magnitude_hist)

        # Unpack paramters (Mu as vector and Sigma as square matrix):
        Mu_final, Sigma_final = BBVI.unstack_params(params, to_square=True)
        Mu_final = Mu_final[0]  # Remove first dimension.
        Sigma_final = Sigma_final[0]  # Remove first dimension.

        return Mu_final, Sigma_final

    @staticmethod
    def stack_params(Mu, Sigma, from_square=False):
        """
        Turns Mu and Sigma into a 1-by-2*D matrix of parameters
        where the covariances are converted to log standard deviations.
        from_square :
            If True, expects Sigma as a diagonal (square) matrix (off-diagonal entries are assumed to be zero).
            If False, expects a vector of the diagonal entries of the covariance matrix.
        """
        if from_square:
            Sigma = np.diagonal(a=Sigma, offset=0, axis1=-1, axis2=-2)
        logStDev = 0.5*np.log(Sigma)
        params = np.concatenate((Mu,logStDev), axis=-1)
        return params

    @staticmethod
    def unstack_params(params, to_square=False):
        """
        Extracts Mu and logStDev from a 1-by-2*D matrix of parameters
        and converts the log standard deviations back to covariances.
        to_square :
            If True, expects Sigma as a diagonal (square) matrix.
            If False, expects a vector of its diagonal entries.
        """
        Mu, logStDev = np.split(params, indices_or_sections=2, axis=-1)
        Sigma = np.exp(2*logStDev)
        if to_square:
            Sigma = np.apply_along_axis(func1d=np.diag, arr=Sigma, axis=-1)
        return Mu, Sigma
