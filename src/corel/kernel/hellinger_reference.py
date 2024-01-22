from typing import Optional

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.kernels import Kernel
from gpflow.utilities import positive, print_summary, to_default_float


class HellingerReference(Kernel):
    def __init__(self, L:int, AA:int, lengthscale: float=1.0, active_dims: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(active_dims, name)
        self.AA = AA
        self.L = L
        self.lengthscale = gpflow.Parameter(lengthscale, transform=positive()) # TODO: log transform here?

    def restore(self, ps: tf.Tensor) -> tf.Tensor:
        if len(ps.shape) == 3: # batch case: [B, N, D]
            return tf.reshape(ps, shape=(ps.shape[1], ps.shape[-1] // self.AA, self.AA))
        return tf.reshape(ps, shape=(ps.shape[0], ps.shape[-1] // self.AA,  self.AA)) # default [N, D]

    def K(self, X, X2=None) -> tf.Tensor:
        if X2 is None:
            X2 = X
        # NOTE: X here are P(X)
        M = self._hellinger2(X, X2)
        M = tf.math.exp(-tf.math.sqrt(M) / tf.math.square(self.lengthscale))
        return M

    def K_diag(self, X) -> tf.Tensor:
        return to_default_float(tf.ones(X.shape[0]))

    def _assert_X_values(self, X: tf.Tensor, tol:float) -> bool:
        return (tf.abs(tf.math.reduce_sum(X, axis=-1) - 1.) < tol).numpy().all()

    def _hellinger2(self, X: tf.Tensor, X2: tf.Tensor, tol: float=1e-5):
        M = np.zeros([X.shape[0], X2.shape[0]], dtype=np.float64)
        X = self.restore(X)
        X2 = self.restore(X2)
        assert self._assert_X_values(X, tol)
        assert self._assert_X_values(X2, tol)
        _tmp = np.zeros(self.L, dtype=np.float64)
        for x_idx in range(X.shape[0]):
            for y_idx in range(X2.shape[0]):
                for l_idx in range(self.L):
                    _tmp[l_idx] = tf.math.reduce_sum(tf.math.sqrt(X[x_idx, l_idx, :] * X2[y_idx, l_idx, :])).numpy() # NOTE: this type of assignment not supported by TF only numpy
                M[x_idx, y_idx] = (1 - tf.reduce_prod(_tmp)).numpy() # NOTE: build np matrix by assignment
        return tf.convert_to_tensor(M)