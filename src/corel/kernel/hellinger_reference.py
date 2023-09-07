from typing import Optional
import tensorflow as tf

import gpflow
from gpflow.kernels import Kernel
from gpflow.utilities import positive
from gpflow.utilities import print_summary


class HellingerReference(Kernel):
    def __init__(self, L:int, AA:int, lengthscale: float=1.0, noise: float=0.1, active_dims: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(active_dims, name)
        self.AA = AA
        self.L = L
        self.lengthscale = gpflow.Parameter(lengthscale, transform=positive()) # TODO: log transform here?
        self.noise = gpflow.Parameter(noise, transform=positive()) # TODO: check against Kernel Interface

    def restore(self, ps: tf.Tensor) -> tf.Tensor:
        ps = tf.squeeze(ps)
        N = 1 if len(ps.shape) == 1 else ps.shape[0]
        return tf.reshape(ps, shape=(N, ps.shape[-1] // self.AA,  self.AA))

    def K(self, X, X2=None) -> tf.Tensor:
        if X2 is None:
            X2 = X
        # NOTE: X here are P(X)
        M = self._hellinger2(X, X2)
        M = tf.math.exp(-tf.math.sqrt(M) / tf.math.square(self.lengthscale))
        return M

    def K_diag(self, X) -> tf.Tensor:
        return tf.ones(X.shape[0])

    def _assert_X_values(self, X: tf.Tensor, tol:float) -> bool:
        return tf.all(tf.abs(tf.math.reduce_sum(X, axis=-1) - 1.) < tol)

    def _hellinger2(self, X: tf.Tensor, X2: tf.Tensor, tol: float=1e-5):
        M = tf.zeros([X.shape[0], X2.shape[0]])
        X = self.restore(X)
        X2 = self.restore(X2)
        assert self._assert_X_values(X, tol)
        assert self._assert_X_values(X2, tol)
        _tmp = tf.zeros(self.L)
        for x_idx in range(X.shape[0]):
            for y_idx in range(X2.shape[0]):
                for l_idx in range(self.L):
                    _tmp[l_idx] = tf.math.reduce_sum(tf.math.sqrt(X[x_idx, l_idx, :] * X2[y_idx, l_idx, :]))
                M[x_idx, y_idx] = 1 - tf.reduce_prod(_tmp)
        return M