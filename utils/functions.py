from autograd import numpy as np
from autograd import scipy as sp

def sigmoid(x):
    """Sigmoid function."""
    return 1.0 / (1 + np.exp(-x))

def bernoulli(k,p):
    """PMF of a Bernoulli distribution."""
    epsilon = 1e-16  # Clip values between epsilon and 1-epsilon.
    p = np.clip(p, a_min=epsilon, a_max=1.0-epsilon)
    result =  (p)**(k) * (1-p)**(1-k)
    return result

def log_bernoulli(k,p, clip=False):
    """Log of the PMF of a Bernoulli distribution."""
    if clip:
        epsilon = 1e-16  # Clip values between epsilon and 1-epsilon.
        p = np.clip(p, a_min=epsilon, a_max=1.0-epsilon)
    result = (k)*np.log(p) + (1-k)*np.log(1-p)
    #result = np.where( k==1, np.log(p) , np.log(1-p) )
    return result

def gaussian(x, mu=0.0, sigma=1.0):
    """PDF of the univariate Gaussian distribution."""
    return 1.0/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x-mu)**2/(2*sigma**2))

def log_gaussian(x, mu=0.0, sigma=1.0):
    """Log PDF of the univariate Gaussian distribution."""
    return - 0.5 * np.log(2*np.pi*sigma**2) - (x-mu)**2/(2*sigma**2)
