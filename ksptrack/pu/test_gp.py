#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)


def f(x, x0=0.5, s=0.1):
    """The function to predict."""

    return np.exp(-0.5 * ((x - x0) / s)**2)
    # return np.sin(x) + x0


X = np.linspace(0., 1., 120)
X = np.atleast_2d(X).T

# Observations and noise
y = f(X).ravel()
dy = 0.1 * np.random.random(X.shape[0]) - 0.2
y += dy

# Instantiate a Gaussian Process model
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.2,
                              n_restarts_optimizer=10)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
x = X
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x)$')
plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate(
             [y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5,
         fc='b',
         ec='None',
         label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.grid()
plt.legend(loc='upper left')

plt.show()
