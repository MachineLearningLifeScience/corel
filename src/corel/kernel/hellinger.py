from typing import Optional
import numpy as np

import tensorflow as tf

import gpflow
from gpflow.kernels import Kernel
from gpflow.utilities import positive
from gpflow.utilities import print_summary


class Hellinger(Kernel):
    def __init__(self, variance: float=1.0, lengthscale: float=1.0, noise: float=0.1, active_dims: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(active_dims, name)
        self.variance = gpflow.Parameter(variance, transform=positive())
        self.lengthscales = gpflow.Parameter(lengthscale, transform=positive()) # TODO: log transform here?
        self.noise = gpflow.Parameter(noise, transform=positive()) # TODO: check against Kernel Interface

    def K(self, X, X2=None) -> tf.Tensor:
        if X2 is None:
            X2 = X
        # NOTE: X here are p-densities
        hd = _hellinger_distance(X)
        lls = tf.log(self.lengthscales)
        lln = tf.log(self.noise)
        return self.variance * _k(hd, lls, lln)

    def K_diag(self, X) -> tf.Tensor:
        return self.variance * tf.reshape(X, (-1,))


def _hellinger_distance(ps):
    """
    This function assumes that elements with exactly the same probability are the same!
    :param ps:
    :type ps:
    :return:
    :rtype:
    """
    squared_hellinger_distance = (ps + tf.transpose(ps)) / 2
    # if we have the same point twice, set the distance to 0 there
    squared_hellinger_distance = tf.where(
        ps - tf.transpose(ps) == 0,
        tf.zeros_like(squared_hellinger_distance), squared_hellinger_distance)
    return squared_hellinger_distance


def _k(HD, log_lengthscale, log_noise):
    K = tf.math.exp(-tf.sqrt(HD) / tf.exp(log_lengthscale))
    K = K + tf.math.exp(log_noise) * tf.eye(K.shape[0], dtype=K.dtype)
    return K


def get_mean_and_amplitude(L, Y):
    """
    This function computes the prior mean constant and the prior amplitude as recommended in the efficient Global
    optimization paper by Jones et al. (1998).
    :param L:
        Cholesky of the kernel matrix
    :param Y:
        the target values
    :return:
        mean constant and amplitude
    """
    ones = tf.linalg.triangular_solve(L, tf.ones_like(Y))
    alpha = tf.linalg.triangular_solve(L, Y)
    n = tf.reduce_sum(tf.square(ones))
    m = tf.reduce_sum(ones * alpha) / n
    #r = tf.reduce_sum(tf.square(alpha - m * tf.ones_like(Y)))
    # Above amplitude estimator seems to be missing a normalization!
    # below is the factor as set in Jones et al. 1998
    r = tf.reduce_sum(tf.square(alpha - m * tf.ones_like(Y))) / Y.shape[0]
    return m, r


if __name__ == "__main__":
    k_hell = Hellinger()
    print_summary(k_hell)