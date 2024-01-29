from typing import Optional

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.kernels import Kernel
from gpflow.utilities import positive, print_summary, to_default_float

from corel.kernel.hellinger_reference import HellingerReference
from corel.util.util import handle_batch_shape


class Hellinger(HellingerReference):
    def __init__(self, L: int, AA: int, lengthscale: float=1.0, active_dims: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(L=L, AA=AA, active_dims=active_dims, name=name)
        self.lengthscale = gpflow.Parameter(lengthscale, transform=positive()) # TODO: log transform here?

    def K(self, X, X2=None) -> tf.Tensor:
        """
        X input is P(X)
        """
        _X = handle_batch_shape(X) # TODO: correct batch handling
        if X2 is None:
            X2 = X
            _X2 = _X
        else:
            _X2 = handle_batch_shape(X2)
        _X = self.restore(X)
        _X2 = self.restore(X2)
        assert _X.shape[-1] == self.AA and _X.shape[-2] == self.L, "Input vector X last two dimensions not consistent! (L, AA)"
        assert _X2.shape[-1] == self.AA and _X2.shape[-2] == self.L, "Input vector X2 last two dimensions not consistent! (L, AA)"

        M = self._H(_X, _X2)

        if len(X.shape) >= 3 or len(X2.shape) >= 3:
            if X is X2:
                M = tf.reshape(M, shape=(1, *M.shape)) # TODO: check shapes and correct
            else:
                M = tf.reshape(M, shape=(1, M.shape[0], 1, M.shape[1])) # adhere to [batch..., N1, batch..., N2]
        return M

    def K_diag(self, X) -> tf.Tensor:
        return to_default_float(tf.ones(X.shape[0]))

    def _H(self, X: tf.Tensor, X2: tf.Tensor):
        M = self._get_inner_product(X, X2)
        M = 1 - M
        M = tf.where(M < 0., tf.zeros_like(M), M)
        
        M = tf.where(M == 0., tf.zeros_like(M), M) # fix gradients
        M = tf.exp(-tf.sqrt(M) / tf.square(self.lengthscale))
        return M

    def _get_inner_product(self, X: tf.Tensor, X2: tf.Tensor):
        # M = tf.math.reduce_sum(tf.sqrt(X[None, ...] * X2[:, None, ...]), axis=-1)
        M = tf.einsum('ali,bli->abl', tf.sqrt(X), tf.sqrt(X2)) # NOTE: the einsum and reduce_sum product should be equivalent
        return tf.math.reduce_prod(M, axis=-1) # product over L, positions factorize


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
    r = tf.reduce_sum(tf.square(alpha - m * ones)) / Y.shape[0]
    return m, r


if __name__ == "__main__":
    k_hell = Hellinger()
    print_summary(k_hell)