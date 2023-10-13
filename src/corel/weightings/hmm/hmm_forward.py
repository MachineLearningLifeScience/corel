__author__ = 'Simon Bartels'

import warnings
import tensorflow as tf
import numpy as np


def tf_forward(s0, T, em, y):
    # TODO: this implementation could maybe be numerically more stable by switching to log-values?
    L = y.shape[0]
    N = em.shape[0]
    alpha = tf.reshape(s0, (N, )) * tf.transpose(em[:, y[0]])
    c = tf.reduce_sum(alpha)
    proba = tf.ones_like(c)
    alpha = alpha / c
    for t in range(1, L):
        alpha = tf.squeeze(tf.expand_dims(alpha, 0) @ T) * tf.transpose(em[:, y[t]])
        c = tf.reduce_sum(alpha)
        alpha = alpha / c
        proba = proba * c
    return proba


def forward(s0, T, em, y):
    # TODO: this implementation could maybe be numerically more stable by switching to log-values?
    L = y.shape[0]
    N = em.shape[0]
    # TODO: in this implementation we could save (quite a bit) of memory here!
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


def _forward_instable(s0, T, em, y):
    L = y.shape[0]
    N = em.shape[0]
    Alpha = np.zeros((L, N))
    Alpha[0, :] = s0.reshape((N, )) * em[:, y[0]].T
    for t in range(1, L):
        Alpha[t, :] = (Alpha[t-1, :] @ T) * em[:, y[t]].T

    # At this point Alpha contains p(x[t]=i,y[:t])
    c = np.sum(Alpha, axis=1)
    # Alpha = np.diag(1. / c) @ Alpha
    # We are supposed to return p(y[t]|y[:t])
    c[1:] /= c[:-1]
    return Alpha, c.reshape(-1, 1)
