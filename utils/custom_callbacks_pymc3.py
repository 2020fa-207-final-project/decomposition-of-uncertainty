"""
Custom Functions for pycm3 callback logging with wandb
"""

from autograd import numpy as np
from autograd import grad
from autograd.misc.optimizers import adam, sgd
from autograd import scipy as sp
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import sys
# import wandb
import pymc3 as pm
import theano.tensor as tt

def plot95ci(nn, X, trace_w):
    # Make predicitons:
    X = np.array(X).flatten().reshape(-1,1)
    N = X.shape[0]  # Number of data points.
    S = trace_w.shape[0]  # Number of models.
    D = trace_w.shape[-1]  # Number of weights.
    W = trace_w.reshape(S,D)  # Weights of S models.
    y_preds = nn.forward(X, W).reshape(S,N)  # Preds for S models.
    # Calculate percentiles
    y_lower = np.percentile(y_preds, q=2.5, axis=0).flatten()
    y_upper = np.percentile(y_preds, q=97.5, axis=0).flatten()
    y_med = np.percentile(y_preds, q=50, axis=0).flatten()
    x_vals = X.flatten()
    # Plot with confidence
    fig, ax = plt.subplots(1,1,figsize=(14,7))
    data = pd.DataFrame({'x_vals':x_vals,'y_lower':y_lower,'y_med':y_med,'y_upper':y_upper})
    ax.plot('x_vals', 'y_upper', data=data, label="Upper 95%", color='grey')
    ax.plot('x_vals', 'y_med', data=data, label="Median Prediction", color='black')
    ax.plot('x_vals', 'y_lower', data=data, label="Lower 95%", color='grey')
    #ax.fill_between('x_vals', 'y_lower', 'y_upper', data=data, alpha=0.4, color='r', label="95% Predictive Interval")
    ax.set_title("Bayesian Neural Net Predictions with 95% CI")
    ax.set_xlabel("X Test")
    ax.set_ylabel("Y Predicted")
    #ax.legend()
    # plt.ylim([-8,4])
    # plt.show()
    return fig, ax

def wb_scatter(nn, X, trace_w):
    X = np.array(X).flatten().reshape(-1,1)
    S = trace_w.shape[0]
    D = trace_w.shape[-1]
    W = trace_w.reshape(S,D)
    y_pred = nn.forward(X, W)
    x_vals = np.tile(X, reps=(S,1,1))
    assert y_pred.shape == x_vals.shape
    x_vals = x_vals.flatten()
    y_pred = y_pred.flatten()
    # Build W&B table and plot:
    data = [[x, y] for (x, y) in zip(x_vals.flatten(), y_pred.flatten())]
    table = wandb.Table(data=data, columns = ["x_test", "y_pred"])
    plot = wandb.plot.scatter(table, "x_test", "y_pred")
    wandb.log({"post_pred_scatter" : plot})
    print(f"Callback: Built plot with {S} samples.")

def build_wb_callback(nn, X, iters_log=25, iters_plot=250):
    it = 0
    X = np.array(X).flatten().reshape(-1,1)
    def wb_callback(trace, draw):
        # Get iteration:
        nonlocal it
        it = it + 1
        if it % iters_log == 0:
            # Log iteration:
            #wandb.log({'iteration' : it}, step=it)
            # Log stats:
            wandb.log(draw.stats[-1], step=it)
        if it % iters_plot == 0:
            # Make plot (optional):
            trace_w = trace.get_values('w')
            fig, ax = plot95ci(nn, X, trace_w)
            wandb.log({"post_pred_95ci": fig}, step=it)
        return
    return wb_callback
