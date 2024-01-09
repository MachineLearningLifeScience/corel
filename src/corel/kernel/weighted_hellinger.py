import gpflow
from gpflow.utilities import positive
import tensorflow as tf
from typing import Optional
import numpy as np

from corel.util.util import handle_batch_shape
from corel.kernel import Hellinger


class WeightedHellinger(Hellinger):
    def __init__(self, w: tf.Tensor, L: int, AA: int, lengthscale: float=1.0, noise: float=0.1, active_dims: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(L=L, AA=AA, active_dims=active_dims, name=name)
        self.w = w  # weighting density vector
        self.lengthscale = gpflow.Parameter(lengthscale, transform=positive()) # TODO: log transform here?
        self.noise = gpflow.Parameter(noise, transform=positive()) # TODO: check against Kernel Interface

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor]=None) -> tf.Tensor:
        """
        X input is P(X)
        """
        _X = handle_batch_shape(X) # TODO: correct batch handling
        if X2 is None:
            X2 = X
            _X2 = _X
        if len(X.shape) == 3: # if [1; N ; D] => [1; N ; L ; cat]
            _X = self.restore(X)
        if len(X2.shape) == 3:
            _X2 = self.restore(X2)
        assert _X.shape[-1] == self.AA and _X.shape[-2] == self.L
        assert _X2.shape[-1] == self.AA and _X2.shape[-2] == self.L

        M = self._H(_X, _X2)

        if len(X.shape) >= 3 or len(X2.shape) >= 3:
            if X.shape == X2.shape:
                M = tf.reshape(M, shape=(1, *M.shape)) # TODO: check shapes and correct
            else:
                M = tf.reshape(M, shape=(1, M.shape[0], 1, M.shape[1])) # adhere to [batch..., N1, batch..., N2]
        return M

    def _get_inner_product(self, X: tf.Tensor, X2: tf.Tensor) -> tf.Tensor:
        """
        Compute RHS of weighted HK equation, as weighting times sqrt(p[a_l,l] x q[a_l,l])
        """
        # M = tf.math.reduce_sum(self.w * tf.sqrt(X[None, ...] * X2[:, None, ...]), axis=-1)
        # NOTE: the einsum and reduce_sum product should be equivalent
        M = tf.einsum('ali,bli->abl', tf.sqrt(tf.pow(self.w,2)*X), tf.sqrt(X2)) 
        return tf.math.reduce_prod(M, axis=-1) # product over L, positions factorize

    def _compute_lhs(self, X: tf.Tensor, X2: tf.Tensor) -> tf.Tensor:
        w_p = tf.math.reduce_sum(self.w*X[None, ...], axis=-1) / 2
        w_q = tf.math.reduce_sum(self.w*X2[:, None, ...], axis=-1) / 2
        return tf.math.reduce_prod(w_p+w_q, axis=-1)

    def _H(self, X: tf.Tensor, X2: tf.Tensor) -> tf.Tensor:
        M = self._get_inner_product(X, X2)
        # NOTE: LHS is expectation with equal weight, could have weighting 
        weighted_E = self._compute_lhs(X, X2)
        M = weighted_E - M 
        M = tf.where(M < 0., tf.zeros_like(M), M)
        M = tf.where(M == 0., tf.zeros_like(M), M) # fix gradients
        M = tf.exp(-tf.sqrt(M) / tf.square(self.lengthscale))
        return M
