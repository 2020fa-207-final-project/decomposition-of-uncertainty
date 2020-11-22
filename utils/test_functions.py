import pytest

from autograd import numpy as np
from autograd import scipy as sp
from scipy import stats  # Non-autograd version.

from utils.functions import *


def test_sigmoid():

    for x in [-55,-33,-4,-.04,0,0.5,3,55,2354]:
    
        assert np.isclose(
            sp.special.expit(x),
            sigmoid(x=x)
        )

def test_bernoulli():

    for p in [0.04,0.44,0.63,0.88,0.99,1.0,0.0]:
    
        assert np.isclose(
            stats.bernoulli.pmf(k=0, p=p),
            bernoulli(k=0, p=p)
        )
    
        assert np.isclose(
            stats.bernoulli.pmf(k=1, p=p),
            bernoulli(k=1, p=p)
        )

        if (p != 0) and (p != 1):
    
            assert np.isclose(
                np.log( stats.bernoulli.pmf(k=0, p=p) ),
                log_bernoulli(k=0, p=p)
            )
        
            assert np.isclose(
                np.log( stats.bernoulli.pmf(k=1, p=p) ),
                log_bernoulli(k=1, p=p)
            )


def test_gaussian():
    
    for mu,sigma in [(0,1),(33,42),(-55,0.0001)]:
        for x in [-55,-33,-4,-.04,0,0.5,3,55,2354]:
            # Check gaussian:
            assert np.isclose(
                sp.stats.norm.pdf(x=x, loc=mu, scale=sigma),
                gaussian(x=x,mu=mu,sigma=sigma)
            )
            # Check log_gaussian:
            assert np.isclose(
                sp.stats.norm.logpdf(x=x, loc=mu, scale=sigma),
                log_gaussian(x=x,mu=mu,sigma=sigma)
            )

