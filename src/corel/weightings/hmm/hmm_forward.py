__author__ = 'Simon Bartels'

import numpy as np


def forward(s0, T, em, y):
    # TODO: this implementation could maybe be numerically more stable by switching to log-values?
    L = y.shape[0]
    N = em.shape[0]
    Alpha = np.zeros((L, N))
    c = np.zeros(L)
    Alpha[0, :] = s0.reshape((N, )) * em[:, y[0]].T
    c[0] = np.sum(Alpha[0, :])
    Alpha[0, :] /= c[0]
    for t in range(1, L):
        Alpha[t, :] = (Alpha[t-1, :] @ T) * em[:, y[t]].T
        c[t] = np.sum(Alpha[t, :])
        Alpha[t, :] /= c[t]
    return Alpha, c.reshape(-1, 1)
