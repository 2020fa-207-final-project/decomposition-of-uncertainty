import pytest

from autograd import numpy as np
from autograd import scipy as sp

from utils.functions import *

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
                log_gaussian(x=x,mu=mu,sigma=sigma) + 1
            )

