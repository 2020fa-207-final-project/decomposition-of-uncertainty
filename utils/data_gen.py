import numpy as np

def sample_gaussian_mixture(N, means, covs, weights):
    '''
    Sample data from a mixture of Gaussians

    Parameters:
        N : int - size of the sample
        means : list - mean value of each gaussian
        covs : covariances of each gaussian
        weights : mixture weights for each gaussian

    Returns:
        mixture_sample : N samples from the mixture of Gaussians
    '''
    # Sample from a multinomial to find the N for each Gaussian
    Z = np.random.multinomial(n=N, pvals=weights)

    # Create samples from each mixture z
    samples = []
    for mean, cov, z in zip(means, covs, Z):
        samples.append(np.random.normal(mean, np.sqrt(cov), z))

    # Flatten the resulting list and return as np.array
    mixture_samples = np.array([sample for subset in samples for sample in subset])
    return mixture_samples


def generate_regression_outputs(type='hsc', N=None, X=None):
    '''
    Generate dummy regression data for testing the BNN+LV

    Parameters:
        type : string - 'hsc' or 'bimodal' (hsc = heteroscedastic)
        N : number of samples - defaults to 750 or len(X) if an X array is passed
        X : np.array - X data used to generate Y data, if not provided defaults 
                       to those provided in the paper for each type

    Returns:
        data_tuple : (Y, X)
    '''
    # Set up N
    if N is None:
        if X is None:
            N = 750
        else:
            N = X.shape[0]

    if type=='hsc':
        # Functional form of the true relationship
        eqn = lambda x: 7 * np.sin(x) + 3 * abs(np.cos(x/2)) * np.random.normal(0,1)
        eqn = np.vectorize(eqn)

        # If no X data is passed then generate it
        if X is None:
            X = sample_gaussian_mixture(N, means=[-4,0,4], covs=[(2/5)**2, 0.9**2, (2/5)**2], weights=[1/3, 1/3, 1/3])

        # Sample the Y data
        Y = eqn(X)

    elif type=='bimodal':
        # Functional form of the true relationship and a bernoulli variable to determine mode
        eqn1 = lambda x, z: z * (10 * np.cos(x) + np.random.normal(0,1)) + (1-z) * (10 * np.sin(x) + np.random.normal(0,1))
        eqn1 = np.vectorize(eqn1)
        Z = np.random.binomial(1,0.5, size=N)

        # If no X data is passed then generate it
        if X is None:
            X = np.random.exponential(1/2, size=N)

            # Rescale to bound between [-0.5, 2] (exponential is from 0 to inf)
            X = (X / (np.max(X)/2.5)) - 0.5

        Y = eqn1(X, Z)
        
    else:
        raise ValueError("Error: type must be one of 'hsc' or 'bimodal'")

    return (Y,X)