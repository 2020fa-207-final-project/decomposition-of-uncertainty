import json
import os
import time

from PIL import Image

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
            callback: A callback function that is run at every wandb checkpoint (e.g. for drawing plots).
                The function expects the HMC instance as its only parameter.
                (Note that it can access the `.base_path` property.)
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
            # Define filename and directory for saving samples/state:
            # Note: By default, wb_base_path is a local folder created automatically by W&B for this run
            #       but for some reason this causes issues when running on DeepNote,
            #       so the use can also specify some other local folder in wb_settings.
            self.wb_base_path = wandb.run.dir if 'base_path' not in self.wb_settings else self.wb_settings['base_path']
            # Save HMC samples/state:
            self.wb_filename = "hmc_state.json" if 'filename' not in self.wb_settings else self.wb_settings['filename']
            self.wb_filepath = os.path.join(self.wb_base_path, self.wb_filename)
            # Bind the W&B module to the instance, for use in callback (e.g. for plotting progress):
            self.wandb = wandb
        
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
        self.n_accepted = None  # Number of accepted samples.
        self.n_rejected = None  # Number of rejected samples.
        self.seconds = None  # Runtime in seconds.
        self.np_random = None  # Numpy random state.

        # Initalize:
        self._reset(warm_start=False)

    def _reset(self, warm_start=False):

        if warm_start:

            pass

        else:

            # Reset state history:
            self.raw_samples = list()
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
                try:
                    self.save_state(self.wb_filepath, replace=True)  # Saves a json file locally.
                    wandb.save(self.wb_filepath, base_path=self.wb_base_path)  # Uploads the file to W&B.
                except Exception as e:
                    print(f"Failed to save {self.wb_filepath} at step {i}.\n\t{e}")
                # Callback function (for producing diagnostic plots):
                callback = None if 'callback' not in self.wb_settings else self.wb_settings['callback']
                if callback is not None:
                    try:
                        if isinstance(callback, list):
                            for func in callback:
                                func(self)  # Pass the HMC object to the callback function.
                        else:
                            callback(self)  # Pass the HMC object to the callback function.
                    except Exception as e:
                        print(f"Callback failed at step {i}.\n\t{e}")

        if self.wb_settings:
            # Save final state:
            try:
                self.save_state(self.wb_filepath, replace=True)  # Saves a json file locally.
                wandb.save(self.wb_filepath, base_path=self.wb_base_path)  # Uploads the file to W&B.
            except Exception as e:
                print(f"Failed to save {self.wb_filepath} at step {i}.\n\t{e}")
            # Finish run:
            try:
                wandb.run.finish()
            except:
                print("W & B run already ended. (Should only happen if .sample() is called more than once.)")
        
        # Stop timer:
        time_end = time.time()
        seconds = time_end-time_start
        if progress:
            print("Finished in {:,} seconds".format(int(seconds)))
        self.seconds += seconds  # Increment timer.

        # Return new samples:
        return self.samples

    def get_samples(self):
        """
        Remove burn-in (by negative indexing) and perform thinning (by list slicing).
        Builds `.samples` from `.raw_samples`.
        """
        return self.raw_samples[self.burn_num::self.thinning_factor]

    @property
    def samples(self):
        return self.get_samples()

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
        self.raw_samples = hmc_state['raw_samples']
        # Get random state:
        random_state = hmc_state['random_state']
        random_state[1] = np.array(random_state[1])
        random_state = tuple(random_state)
        self.np_random.set_state(random_state)
        print(f"Loaded HMC state : {filepath} .")


def build_wb_callback_plotfunc(plot_func,filename='posterior_predictive',**kwargs):
    """
    Wrap a plotting function to produce plots for logging to Weights & Biases during HMC sampling.
    Assumes the function takes a `samples` argument (S by D array) and returns pyplot axes.
    The rest of the keyword arguments are passed directly to the plotting function.
    """
    assert 'samples' not in kwargs, "No need to provide `samples`, as they will we extract from current HMC state."
    def callback(sampler):
        filepath = os.path.join(sampler.wb_base_path, filename+'.png')
        if len(sampler.samples)>0:
            samples = sampler.get_samples()  # Get samples from sampler.
            samples = np.vstack(samples)  # Convert to numpy array.
            kwargs['samples'] = samples  # Add samples to keyword arguments.
            ax = plot_func(**kwargs)  # Call plotting function.
            ax.figure.savefig(filepath)  # Save plot locally.
            sampler.wandb.save(filepath, base_path=sampler.wb_base_path)  # Upload plot file to W&B.
            img = Image.open(filepath)  # Load image as array.
            sampler.wandb.log({filename:[sampler.wandb.Image(img, caption=filename)]})
            print(f"Callback: Saved plot {filepath} ({samples.shape[0]} samples).")
        else:
            print(f"Callback: No samples to plot.")
    return callback


def build_wb_callback_postpred(model, x_data):
    """
    Build a callback function that produces a scatterplot of the posterior predictive
    for Weights and Biases.
    model:
        A model with a `.forward(X, weights)` method.
    x_data:
        The X values to plot (vector of length N; or )
    """
    x_data = np.array(x_data).flatten().reshape(-1,1)
    def wb_post_pred(sampler):
        if len(sampler.samples)>0:
            samples = sampler.get_samples()  # Get samples from sampler.
            samples = np.vstack(samples)  # Convert to numpy array.
            S = samples.shape[0]  # Get number of models.
            y_pred = model.forward(x_data, samples)
            x_vals = np.tile(x_data, reps=(S,1,1))
            assert y_pred.shape == x_vals.shape
            # Build W&B table and plot:
            data = [[x, y] for (x, y) in zip(x_vals.flatten(), y_pred.flatten())]
            table = sampler.wandb.Table(data=data, columns = ["class_x", "class_y"])
            sampler.wandb.log({"posterior_predictic" : sampler.wandb.plot.scatter(table, "class_x", "class_y")})
            print(f"Callback: Built plot with {samples.shape[0]} samples.")
        else:
            print(f"Callback: No samples to plot.")
    return wb_post_pred
    

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
        Mu_init_Z=None, Sigma_init_Z=None,
        num_samples=1_000, step_size=0.1, num_iters=1_000,
        random_seed=None, progress=False,
        wb_settings=False,
    ):
        """
        Initialize BBVI.

        log_target_func :
            The function that is being approximated.
        
        Mu_init :
            Initialization value for the mean of the variational distribution of W.
            Expects Mu as a length-D vector or 1-by-D matrix.
        
        Sigma_init :
            Initialization value for the covariance of the variational distribution of Z.
            Expects Sigma as a legnth-D vector, 1-by-D matrix, or D-by-D diagonal matrix (off-diagonal entries are assumed to be zero).
        
        Mu_init_Z :
            Initialization value for the mean of the variational distribution of W.
            Expects Mu_Z as a length-N vector or N-by-L matrix.
        
        Sigma_init_Z :
            Initialization value for the covariance of the variational distribution of Z.
            Expects Sigma_Z as a legnth-N vector or N-by-L matrix.

        mode :
            The type of neural net (determines which ELBO is used).
            Options are: 'BNN', 'BNN_LV'.
        
        num_samples :
            Number of noise samples taken during the reparameterization step.
        
        step_size :
            Step size for gradient descent.
        
        num_iters :
            Number of steps of gradient desecent.
        
        random_seed :
            Seed for a fixed numpy random state (or None).
        
        progress :
            Number of iterations at which to print progress updates (or False).

        wb_settings:
            (False or dict) Settings for logging to Weights & Biases (optional).
            entity: Username of the project host (by default, uses gpestre/am207 shared project).
            project: Name of the project (by default, uses gpestre/am207 shared project).
            group: Name of the group (e.g. bbvi_bnn_lv) 
            name: Short name for this particular run.
            notes: Short note describing the tuning for this particular run.
                (Note: The full dictionary of hyperparameters will also be uploaded.)
            save_code: Whether or not to upload a copy of the script/notebook BBVI was called run from.
                (If True, also requires allowing code uploads in the settings of your W&B account.)
            [The options above are sent to DeepNote's init function -- see here: https://docs.wandb.com/library/init .]
            [The options below are additional parameters for uploading samples and performance metrics to W&B .]
            progress: Integer indicating how often to update progress (e.g. every 1000 steps).
            base_path: Local directory where samples will be saved before W&B upload.
                (If blank, uses the folder wandb creates for this run;
                the default is a good choice, but does not seem to work on DeepNote.)
            filename: Name of the file the BBVI parameters/state are dumped to (default: "bbvi_state.json").
            archive: A static dictionary of params/values to archive (e.g. info about the priors).
                (These are logged as hyperparameters addition to the BBVI intialization parameters).
            callback: A callback function that is run at every wandb checkpoint (e.g. for drawing plots).
                The function expects the BBVI instance as its only parameter.
                (Note that it can access the `.base_path` property.)
        """

        # Initialize W & B logging (optional):
        self.wb_settings = wb_settings
        self.wb_progress = None if (not self.wb_settings) or ('progress' not in wb_settings) else wb_settings['progress']
        if self.wb_settings is not False:
            # Create a dictionary of hyperparameters:
            config = dict() if 'archive' not in wb_settings else wb_settings['archive']
            config.update({
                'mode' : mode,
                'num_samples' : num_samples,
                'step_size' : step_size,
                'num_iters' : num_iters,
                'random_seed' : random_seed,
                'Mu_init' : Mu_init,
                'Sigma_init' : Sigma_init,
                'Mu_init_Z' : Mu_init_Z,
                'Sigma_init_Z' : Sigma_init_Z,
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
                group     = get_wb_setting(key='group', default='bbvi'),
                name      = get_wb_setting(key='name', default='bbvi'),
                save_code = get_wb_setting(key='save_code', default=False),
                notes     = get_wb_setting(key='notes', default=None),
                config    = config,  # Dictionary of the hyperparameters.
            )
            # Define filename and directory for saving samples/state:
            # Note: By default, wb_base_path is a local folder created automatically by W&B for this run
            #       but for some reason this causes issues when running on DeepNote,
            #       so the use can also specify some other local folder in wb_settings.
            self.wb_base_path = wandb.run.dir if 'base_path' not in self.wb_settings else self.wb_settings['base_path']
            # Save HMC samples/state:
            self.wb_filename = "bbvi_state.json" if 'filename' not in self.wb_settings else self.wb_settings['filename']
            self.wb_filepath = os.path.join(self.wb_base_path, self.wb_filename)
            # Bind the W&B module to the instance, for use in callback (e.g. for plotting progress):
            self.wandb = wandb

        # Check mode:
        valid_modes = {'BNN','BNN_LV'}
        mode = mode.replace('+','_').upper()
        assert mode in valid_modes, f"{mode} is not a valid mode: {valid_modes}"
        self.mode = mode

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
            raise ValueError("Expects Sigma as a legnth-D vector, 1-by-D matrix, or D-by-D diagonal matrix (off-diagonal entries are assumed to be zero).")
        
        # Check that Mu and Sigma match:
        assert Mu_init.shape[1]==Sigma_init.shape[1], "Expect Mu and Sigma to have same dimension."
        dims = Mu_init.shape[1]
        # Convert covariance to log standard deviation:
        logStDev_init = 0.5*np.log(Sigma_init)
        
        if self.mode=='BNN_LV':

            # Check dimensions of Mu_Z (and infer number of latent features):
            try:
                if len(Mu_init_Z.shape)==1:
                    # Convert vector of N values to N-by-L matrix (assuming L=1):
                    Mu_init_Z = Mu_init_Z.reshape(-1,1)
                elif len(Mu_init_Z.shape)==2 and Mu_init_Z.shape[0]:
                    # Keep N-by-L matrix:
                    if (Mu_init_Z.shape[0]==1) and (Mu_init_Z.shape[1]!=1):
                        print("NOTE: Mu_init_Z should be N-by-L (where L is often 1), but in this case N=1; is that intentional?")
                else:
                    raise ValueError
            except:
                raise ValueError("Expects Mu_Z as a length-N vector or N-by-L matrix.")

            # Check dimensions of Sigma_Z:
            try:
                if len(Sigma_init_Z.shape)==1:
                    # Convert vector of N values to N-by-L matrix (assuming L=1):
                    Sigma_init_Z = Sigma_init_Z.reshape(-1,1)
                elif len(Sigma_init_Z.shape)==2:
                    # Keep N-by-L matrix:
                    if (Sigma_init_Z.shape[0]==1) and (Sigma_init_Z.shape[1]!=1):
                        print("NOTE: Sigma_init_Z should be N-by-L (where L is often 1), but in this case N=1; is that intentional?")
                elif len(Sigma_init_Z.shape)==3:
                    raise NotImplementedError  # This case is ambiguous.
                else:
                    raise ValueError
            except:
                raise ValueError("Expects Sigma_Z as a legnth-N vector or N-by-L matrix.")

            # Check that Mu and Sigma match:
            assert Mu_init_Z.shape==Sigma_init_Z.shape, f"Expect Mu_Z {Mu_init_Z.shape} and Sigma_Z {Sigma_init_Z.shape} to have same dimension."
            N, L = Mu_init_Z.shape
            # Convert covariance to log standard deviation:
            logStDev_init_Z = 0.5*np.log(Sigma_init_Z)

        elif self.mode!='BNN_LV':

            assert Mu_init_Z is None, "This parameter is only for mode=BNN_LV."
            assert Sigma_init_Z is None, "This parameter is only for mode=BNN_LV."

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

        # Params specific to BNN_LV:
        if self.mode=='BNN_LV':
            self.Mu_init_Z = Mu_init_Z
            self.Sigma_init_Z = Sigma_init_Z
            self.logStDev_init_Z = logStDev_init_Z
            self.dims_Z = N * L  # Size of the `stacked` version the parameters associated with Z.
            self.L = L  # Number of latent features.
            self.N = N  # Number of data points (Note: BBVI estimates a mean and variance for each Z of each data point).
        
        # Represent position as a 1-by-2D matrix:
        if self.mode=='BNN':
            self.params_init = self._stack(Mu=Mu_init, logStDev=logStDev_init)
        elif self.mode=='BNN_LV':
            self.params_init = self._stack(Mu=Mu_init, logStDev=logStDev_init, Mu_Z=Mu_init_Z, logStDev_Z=logStDev_init_Z)

        # Build placeholder for state variables:
        self.params_hist = None  # History of parameters at each interation (built as list of arrays and converted to numpy 2D array).
        self.gradident_hist = None  # History of gradient at each iteration (built as list of arrays and converted to numpy 2D array).
        self.elbo_hist = None  # History of ELBO value at each iteration (built as list and converted to numpy 1D array).
        self.magnitude_hist = None  # History of gradient magnitude at each iteration (built as list and converted to numpy 1D array).
        self.seconds = None  # Runtime in seconds.
        self.np_random = None  # Numpy random state.
        self.variational_gradient = None  # Gradient function (computed with autograd).
        
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

    def _stack(self, Mu, logStDev, Mu_Z=None, logStDev_Z=None):
        """
        Turns Mu and logStDev into a 1-by-2*D matrix of parameters.
        (This is the interal version of the function that uses flat vectors and log-standard-devations;
        a user-facing version that uses covariance and (optionally) square matrics is provided as a class method.)
        """
        if self.mode=='BNN':
            params = np.concatenate([Mu,logStDev], axis=-1)
            return params
        elif self.mode=='BNN_LV':
            Mu_Z = Mu_Z.reshape(1,-1)  # Flatten N-by-L matrix into 1-by-(N*L) vector.
            logStDev_Z = logStDev_Z.reshape(1,-1)  # Flatten N-by-L matrix into 1-by-(N*L) vector.
            params = np.concatenate([Mu,logStDev,Mu_Z,logStDev_Z], axis=-1)  # Result is 1-by-2*(D+N*L)
            return params
        else:
            raise NotImplementedError(f"The _stack method is not yet implemented for mode {self.mode}.")
        
    def _unstack(self, params):
        """
        Extracts Mu and logStDev from a 1-by-2*D matrix of parameters.
        (This is the interal version of the function that uses flat vectors and log-standard-devations;
        a user-facing version that uses covariance and (optionally) square matrics is provided as a class method.)
        """
        if self.mode=='BNN':
            Mu = params[:,:self.dims]
            logStDev = params[:,self.dims:]
            return Mu, logStDev
        elif self.mode=='BNN_LV':
            Mu = params[:,:self.dims]
            logStDev = params[:,self.dims:(2*self.dims)]
            Mu_Z = params[:,(2*self.dims):(2*self.dims+self.dims_Z)]
            logStDev_Z = params[:,(2*self.dims+self.dims_Z):]
            Mu_Z = Mu_Z.reshape(self.N,self.L)  # Expand 1-by-(N*L) vector back to N-by-L matrix.
            logStDev_Z = logStDev_Z.reshape(self.N,self.L)  # Expand 1-by-(N*L) vector back to N-by-L matrix.
            return Mu, logStDev, Mu_Z, logStDev_Z
        else:
            raise NotImplementedError(f"The _stack method is not yet implemented for mode {self.mode}.")

    # Define optimizer and callback:
    def _callback(self, params, iteration, gradient):
        # Calculate ELBO:
        elbo_value = -self.variational_objective(params, iteration)
        elbo_value = elbo_value[0]  # Unpack single value from matrix.
        # Calculate magnitude of gradient:
        grad_mag = np.linalg.norm(self.variational_gradient(params, iteration))
        # Update history:
        self.params_hist.append(params)
        self.gradident_hist.append(gradient)
        self.elbo_hist.append(elbo_value)
        self.magnitude_hist.append(grad_mag)
        # Print progress (optional):
        if self.progress and ((iteration+1) % self.progress == 0):
            printout = "Iteration {} : lower bound = {}, gradient magnitude = {}".format(iteration+1, elbo_value, grad_mag)
            print(printout)
            #print("params :",params)
            #print("gradient :",gradient)
        # Log progress to W&B (optional):
        if self.wb_progress and ((iteration+1) % self.wb_progress == 0):
            # Upload performance metrics:
            wandb.log({
                'params' : params,
                'gradient' : gradient,
                'elbo_value' : elbo_value,
                'grad_mag' : grad_mag,
            }, step=iteration+1)
            # Save samples/state and upload them:
            try:
                self.save_state(self.wb_filepath, replace=True)  # Saves a json file locally.
                wandb.save(self.wb_filepath, base_path=self.wb_base_path)  # Uploads the file to W&B.
            except Exception as e:
                print(f"Failed to save {self.wb_filepath} at step {iteration+1}.\n\t{e}")
            # Callback function (for producing diagnostic plots):
            callback = None if 'callback' not in self.wb_settings else self.wb_settings['callback']
            if callback is not None:
                try:
                    if isinstance(callback, list):
                        for func in callback:
                            func(self)  # Pass the HMC object to the callback function.
                    else:
                        callback(self)  # Pass the HMC object to the callback function.
                except Exception as e:
                    print(f"Callback failed at step {iteration+1}.\n\t{e}")

    def gaussian_entropy(self, logStDev):
        """
        Calculates the Gaussian Entropy of logStDev
        (assuming logStDev is diagonal, and passed as a 1-by-D matrix).
        """
        logStDev = logStDev.flatten()  # Where there are multiple latent features, unravel them into a vector.
        dims = len(logStDev)
        # See : https://statproofbook.github.io/P/mvn-dent
        # Note that since the covariance matrix is diagonal: log(det(Covar)) is equivalent to trace(logStDev).
        # Here, logStDev is a vector of the logs of the square roots
        # of the diagonal entries of the covariange matrix.
        # (i.e. logStDev is standard deviations, whereas the typical G.E. formula uses the covariance matrix).
        #dims = logStDev.shape[-1]
        #return dims/2*np.log(2*np.pi) + 1/2*np.sum(logStDev**2,axis=-1) + 1/2*dims
        return 0.5* dims *( 1.0 + np.log(2*np.pi) ) + np.sum(logStDev,axis=-1)

    def variational_objective(self, params, iteration=None):
        """
        Provides a stochastic estimate of the variational lower bound.
        (The `iteration` parameter is required by ADAM but is not used.)
        """
        if self.mode=='BNN':
            Mu, logStDev = self._unstack(params)
            eps_S = self.np_random.randn(self.num_samples,self.dims)  # Each row is a different sample.
            StDev = np.exp(logStDev)
            W_S = eps_S * StDev + Mu  # Perturb StDev element-wise for each of `num_samples` in eps_S.
            posterior_term = np.mean(self.log_target_func(W_S), axis=0)
            gaussian_entropy_term = self.gaussian_entropy(logStDev)
            elbo_approx = posterior_term + gaussian_entropy_term
            return -elbo_approx
        elif self.mode=='BNN_LV':
            Mu, logStDev, Mu_Z, logStDev_Z = self._unstack(params)
            # W part:
            #   Note: We drop the first dimension of the weights, which is 1, to have an S by D result.
            eps_S = self.np_random.randn(self.num_samples,1,self.dims)  # Each row is a different sample.
            StDev = np.exp(logStDev)
            W_S = eps_S * StDev + Mu  # Perturb StDev element-wise for each of `num_samples` in eps_S.
            # Z part:
            #   Note: We keep the first dimension of the weights, which is L, to have an S by L by D result.
            eps_Z_S = self.np_random.randn(self.num_samples,self.N,self.L)  # Each row is a different sample.
            StDev_Z = np.exp(logStDev_Z)
            Z_S = eps_Z_S * StDev_Z + Mu_Z  # Perturb StDev element-wise for each of `num_samples` in eps_S.
            # Joint part:
            posterior_term = np.mean([
                self.log_target_func(w, z)
                for w, z in zip(W_S,Z_S)
            ], axis=0)
            gaussian_entropy_term = self.gaussian_entropy(logStDev) + self.gaussian_entropy(logStDev_Z)
            elbo_approx = posterior_term + gaussian_entropy_term
            # Return is a scalar but return it as a 1-value vector (for stacking):
            elbo_approx = elbo_approx.reshape(1)  # Will (correctly) throw an error if not a scalar.
            return -elbo_approx
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

        # Perform optimization with ADAM:
        params = adam(
            grad = self.variational_gradient,
            x0 = self.params_init,
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

        # Finish W&B logging:
        if self.wb_settings:
            # Save final state:
            try:
                self.save_state(self.wb_filepath, replace=True)  # Saves a json file locally.
                wandb.save(self.wb_filepath, base_path=self.wb_base_path)  # Uploads the file to W&B.
            except Exception as e:
                print(f"Failed to save {self.wb_filepath} at step {i}.\n\t{e}")
            # Finish run:
            try:
                wandb.run.finish()
            except:
                print("W & B run already ended. (Should only happen if .sample() is called more than once.)")
        
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

    @property
    def params(self):
        if len(self.params_hist) == 0:
            return None
        return self.params_hist[-1]

    def get_samples(self, num=None, seed=None):
        # Determine how many samples to get: 
        num_samples = self.num_samples if num is None else num
        # Use sampler's random state, unless otherwise specified:
        np_random = self.np_random if seed is None else np.random.RandomState(seed)
        # Get means and variances of the approximation distributions:
        if self.mode=='BNN':
            Mu, logStDev = self._unstack(self.params)
            Sigma = np.exp(logStDev)**2
            # Use means and variances to generate samples from proposal distributions:
            samples = np_random.normal(loc=Mu, scale=Sigma, size=(num_samples, *Mu.shape))
            return samples
        elif self.mode=='BNN_LV':
            Mu, logStDev, Mu_Z, logStDev_Z = self._unstack(self.params)
            Sigma = np.exp(logStDev)**2
            Sigma_Z = np.exp(logStDev_Z)**2
            # Use means and variances to generate samples from proposal distributions:
            samples = np_random.normal(loc=Mu, scale=Sigma, size=(num_samples, *Mu.shape))
            samples_Z = np_random.normal(loc=Mu_Z, scale=Sigma_Z, size=(num_samples, *Mu_Z.shape))
            return (samples, samples_Z)
        else:
            raise NotImplementedError(f"The _stack method is not yet implemented for mode {self.mode}.")
    
    def save_state(self, filepath, replace=False):
        # Get histories (as lists of lists):
        params_hist = np.array(self.params_hist).tolist()
        gradident_hist = np.array(self.gradident_hist).tolist()
        elbo_hist = np.array(self.elbo_hist).tolist()
        magnitude_hist = np.array(self.magnitude_hist).tolist()
        # Get random state:
        random_state = list(self.np_random.get_state())
        random_state[1] = random_state[1].tolist()
        # Make dictionary:
        bbvi_state = {
            'params_hist' : params_hist,
            'gradident_hist' : gradident_hist,
            'elbo_hist' : elbo_hist,
            'magnitude_hist' : magnitude_hist,
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
            json.dump(bbvi_state, f, indent=4)
        print(f"Saved BBVI state : {filepath} .")
        
    def load_state(self, filepath):
        # Load dictionary:
        with open(filepath, 'r') as f:
            bbvi_state = json.load(f)
        # Get histories:
        self.params_hist = np.vstack(bbvi_state['params_hist'])
        self.gradident_hist = np.vstack(bbvi_state['gradident_hist'])
        self.elbo_hist = np.array(bbvi_state['elbo_hist'])
        self.magnitude_hist = np.array(bbvi_state['magnitude_hist'])
        # Get random state:
        random_state = bbvi_state['random_state']
        random_state[1] = np.array(random_state[1])
        random_state = tuple(random_state)
        self.np_random.set_state(random_state)
        print(f"Loaded BBVI state : {filepath} .")

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
