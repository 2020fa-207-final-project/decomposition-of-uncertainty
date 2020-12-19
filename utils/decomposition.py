import scipy as sp
import autograd.numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform

import matplotlib.pyplot as plt

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

def moving_average(x, w):
    # moving average for simplified uncertainty plotting
    # https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-numpy-scipy
    return np.convolve(x, np.ones(w), 'valid') / w

def uncertainty_decompose_entropy(bnn_lv, X_train, w_samples, S, N, N2, D, avg_window=10):
    """
    S = number of samples
    N  = number of x datapoints
    N2 = number of x datapoints for calculating entropy
    D = Dimensions of y
    """


    x_test_space = np.linspace(min(X_train), max(X_train), N)
    x_test_space_small = np.linspace(min(X_train), max(X_train), N2)
    w_random_samples = w_samples[np.random.choice(w_samples.shape[0], S), :].squeeze(1)

    L = 100 # Number of y points to take per set of samples for epistemic uncertainty

    # Get the stack of predictions from the 2000 samples of weights (over 1000 x data points)
    y_star = bnn_lv.forward(x_test_space_small.reshape(-1,1), w_random_samples)

    # Reshape it to be (1000 x 2000 x 1 - NxSxM - stacks(x) by samples by dimensions) - calculate entropies
    overall_entropy = knn_entropy(y_star.swapaxes(0,1))

    # Create a duplicated set of data to predict L times per set of samples
    x_test_stack = np.tile(x_test_space_small.reshape(-1,1), reps=(1,L)).reshape(-1,1)
    y_stack = bnn_lv.forward(x_test_stack, w_random_samples)
    y_stack = y_stack.reshape(S,-1,L,D)

    epi_H_W = np.stack([knn_entropy(tensor) for tensor in y_stack],axis=-1)
    aleatoric_entropy = np.mean(epi_H_W, axis=-1)

    epistemic_entropy = overall_entropy - aleatoric_entropy


    fig, ax = plt.subplots(1,3,figsize=(21,6))
    ax[0].plot(moving_average(x_test_space_small.flatten(),avg_window),
            moving_average(overall_entropy.flatten(),avg_window))
    ax[0].set_title('Total Uncertainty')

    ax[1].plot(moving_average(x_test_space_small.flatten(),avg_window),
            moving_average(aleatoric_entropy.flatten(),avg_window))
    ax[1].set_title('Aleatoric Uncertainty')

    ax[2].plot(moving_average(x_test_space_small.flatten(),avg_window),
            moving_average(epistemic_entropy.flatten(),avg_window))
    ax[2].set_title('Epistemic Uncertainty')
    plt.show()
    
    return epistemic_entropy, aleatoric_entropy


def chicken_entropy_decompose(bnn_lv_in, transitions_in, w_samples_in, N, N2, L):  
    """
    2D decomposition for 3x5 wet chicken grid
    """
        
    S = w_samples_in.shape[0] # number of samples
    D = 2 # Dimensions of y
      
    x1_test_space_lst = np.arange(min(transitions_in['start_x']), 1+max(transitions_in['start_x']))
    x2_test_space_lst = np.arange(min(transitions_in['start_y']), 1+max(transitions_in['start_y']))

    x1_test_space_grid, x2_test_space_grid = np.meshgrid(x1_test_space_lst,x2_test_space_lst)

    x1_test_space = list(x1_test_space_grid.ravel())*N
    x2_test_space = list(x2_test_space_grid.ravel())*N
    x_test_space = np.vstack([x1_test_space,x2_test_space,
                          [0]*len(x2_test_space), [0]*len(x2_test_space)]).T


    x1_test_space_small = list(x1_test_space_grid.ravel())*N2
    x2_test_space_small = list(x2_test_space_grid.ravel())*N2
    x_test_space_small = np.vstack([x1_test_space_small,x2_test_space_small,
                          [0]*len(x2_test_space_small), [0]*len(x2_test_space_small)]).T

    
    
    # Get the stack of predictions from the S samples of weights
    y_star = bnn_lv_in.forward(x_test_space_small, w_samples_in.squeeze(1))

    # Reshape it to be (NxSx2- stacks(x) by samples by dimensions) - calculate entropies
    overall_entropy = knn_entropy(y_star.swapaxes(0,1))

    # Create a duplicated set of data to predict L times per set of samples
    x_test_stack = np.tile(x_test_space_small, reps=(L,1))
    y_stack = bnn_lv_in.forward(x_test_stack, w_samples_in.squeeze(1))
    y_stack = y_stack.reshape(S,-1,L,D) #samples x N x L x dim

    epi_H_W = np.stack([knn_entropy(tensor) for tensor in y_stack],axis=-1)
    aleatoric_entropy = np.mean(epi_H_W, axis=-1)

    epistemic_entropy = overall_entropy - aleatoric_entropy
    
    ### Saving out dataframe
    decomposition_df = pd.DataFrame({'x':x_test_space_small[:,0], 'y':x_test_space_small[:,1],
                                     'epistemic':epistemic_entropy, 'aleatoric':aleatoric_entropy})
    decomposition_df.sort_values(['x','y'],inplace=True)
    decomposition_df_avg = decomposition_df.groupby(['x','y']).mean().reset_index()
    
    ### Plotting
    
    epistemic_im = np.zeros((5,3))
    aleatoric_im = np.zeros((5,3))
    for i, row in decomposition_df_avg.iterrows():
        x = int(row['x'])-1
        y = int(row['y'])-1
        epistemic_im[y,x] = row['epistemic']
        aleatoric_im[y,x] = row['aleatoric']
    
    fig, ax = plt.subplots(1,2,figsize=(20,10))
    plot1 = ax[0].imshow(epistemic_im,origin='lower')
    ax[0].set_title('Epistemic')
    ax[0].set_xticks([0,1,2])
    ax[0].set_xticklabels([1, 2, 3])
    ax[0].set_xlabel('x')
    ax[0].set_yticks(np.arange(5))
    ax[0].set_yticklabels(np.arange(1,6))
    ax[0].set_ylabel('y')
    for (j,i),label in np.ndenumerate(np.round(epistemic_im,2)):
        ax[0].text(i,j,label,ha='center',va='center')
    fig.colorbar(plot1,ax=ax[0])

    plot2 = ax[1].imshow(aleatoric_im,origin='lower')
    ax[1].set_title('Aleatoric')
    ax[1].set_xticks([0,1,2])
    ax[1].set_xticklabels([1, 2, 3])
    ax[1].set_xlabel('x')
    ax[1].set_yticks(np.arange(5))
    ax[1].set_yticklabels(np.arange(1,6))
    ax[1].set_ylabel('y')
    for (j,i),label in np.ndenumerate(np.round(aleatoric_im,2)):
        ax[1].text(i,j,label,ha='center',va='center')
    fig.colorbar(plot2,ax=ax[1])

    plt.show()
    
    return decomposition_df_avg
    