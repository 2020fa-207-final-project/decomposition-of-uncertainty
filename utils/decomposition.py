import scipy as sp
import autograd.numpy as np

from scipy.spatial.distance import pdist, squareform

## Entropy calculation
def knn_entropy_2D(data_points, metric="euclidean"):
    """
    Inner function to which applies to calculation to each matrix (see knn_entropy for details)
    """
    # Set up constants 
    p = data_points.shape[1]
    n = data_points.shape[0]

    hypersphere_k = np.log( (np.pi ** (p/2)) / sp.special.gamma((p/2) + 1) )
    euler_k = np.euler_gamma + np.log(n-1)

    # Set the distance to self to infinity
    inf_diag = np.diag(np.ones(data_points.shape[0]) * np.inf)

    # Loop over each stack and calculate the entropy
    #for stack in data_points.shape[-1]

    # Create the distance matrix from point to all other points
    distances = squareform(pdist(data_points, metric)) + inf_diag

    # Closest neighbor distance
    p_i = np.min(distances, axis=1)

    # Calculate Entropy
    H = p * np.mean(np.log(p_i)) + hypersphere_k + euler_k

    return H


def knn_entropy(data_points, metric="euclidean"):
    """
    Utilize KNN approximation to calculate the entropy of a set of points.

    Data must be SxNxM where N = data points, M = dimensionality, S = stack 
    (as we are calculating entropy per datapoint S is usually going to be X)

    If distance are all zero (say a vector of ones) then returns -inf
    """
    assert(isinstance(data_points, np.ndarray)), "Error: Data must be a numpy 3D array (rows = data, columns = dimensions, depth=stacks)"
    assert(len(data_points.shape) == 3), "Error: Data must be a numpy 3D array (rows = data, columns = dimensions, depth=stacks)"

    H = np.stack([knn_entropy_2D(matrix) for matrix in data_points],axis=-1)

    return H