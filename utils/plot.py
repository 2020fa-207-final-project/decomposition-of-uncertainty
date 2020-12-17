import matplotlib.pyplot as plt
import autograd.numpy as np

def plot_posterior_predictive(x_data, model, samples, mode='points', ax=None, figsize=(14,7), real_x=None, real_y=None):
    """
    Plot the posterior predictive graph in one of two modes
    """
    assert(mode in ['points', 'fill']), "Error: Mode must be 'points' (scatter) or 'fill' (95% CI)"

    if len(x_data.shape) == 1:
        x_data = x_data.reshape(-1,1)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    y_pred = model.forward(x_data, samples)

    if mode == 'points':
        y_pred_flat = y_pred.reshape(-1)
        x_data_flat = np.repeat(x_data, samples.shape[0], axis=-1).reshape(-1, order='F')

        ax.scatter(x_data_flat, y_pred_flat, color='b', alpha=0.15, s=2, label = 'posterior predictive samples')

    else:
        # Calculate percentiles
        y_lower = np.percentile(y_pred, q=2.5, axis=0)
        y_upper = np.percentile(y_pred, q=97.5, axis=0)
        y_med = np.percentile(y_pred, q=50, axis=0)

        if real_x is not None and real_y is not None:
            ax.scatter(real_x.flatten(), real_y.flatten(), color='black', label='data')

        ax.plot(x_data, y_med, label="Median Prediction")
        ax.fill_between(x_data.reshape(-1), y_lower.reshape(-1), y_upper.reshape(-1), alpha=0.4, color='r', label="95% Predictive Interval")
    
    return ax
    
