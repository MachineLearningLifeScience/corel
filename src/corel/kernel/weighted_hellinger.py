from typing import Optional

import gpflow
import numpy as np
import tensorflow as tf
from gpflow import default_float
from gpflow.utilities import positive

from corel.kernel import Hellinger
from corel.util.util import handle_batch_shape


class WeightedHellinger(Hellinger):
    def __init__(self, w: tf.Tensor, L: int, AA: int, lengthscale: tf.Tensor=None, active_dims: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(L=L, AA=AA, active_dims=active_dims, name=name)
        if len(w.shape) >= 3:
            w = self.restore(w)
        w = tf.squeeze(w)
        assert len(w.shape) == 2, "Weighting vector has too many dimensions!"
        self.w = w  # weighting density vector
        if lengthscale is None:
            lengthscale = gpflow.Parameter(1.0, dtype=default_float()) # TODO: log transform here?
        self.lengthscale = lengthscale  
            
    @staticmethod
    def _handle_k_output_shape(X: tf.Tensor, X2: tf.Tensor) -> tuple:
        if X2 is None:
            if len(X.shape) == 2: # case [N, D] singular
                output_shape = (X.shape[0], X.shape[0])
            elif len(X.shape) == 3: # case [B, N, D] singular
                output_shape = (1, X.shape[1], X.shape[1])
            else:
                raise ValueError(f"The provided input shapes X.shape={X.shape} , X2.shape={X2.shape} are incorrect!\n Required X.shape=[N, D] or X2.shape=[B, N, D]")
        elif len(X.shape) == 2 and len(X2.shape) == 2: # case two inputs of shape [N, D] 
            output_shape = (1, X.shape[0], 1, X2.shape[0]) # adhere to [batch..., N1, batch..., N2]
        elif len(X.shape) == 2 and len(X2.shape) == 3: # case two inputs of shape [N, D] and [B, N, D]
            output_shape = (1, X.shape[0], 1, X2.shape[1]) # adhere to [batch..., N1, batch..., N2]
        elif len(X.shape) == 3 and len(X2.shape) == 2: # case two inputs of shape [B, N, D] and [N, D]
            output_shape = (1, X.shape[1], 1, X2.shape[0]) # adhere to [batch..., N1, batch..., N2]
        elif len(X.shape) == 3 and len(X2.shape) == 3: # case two inputs of shape [B, N, D] 
            output_shape = (1, X.shape[1], 1, X2.shape[1]) # adhere to [batch..., N1, batch..., N2]
        else:
            raise ValueError(f"The provided input shapes X.shape={X.shape} , X2.shape={X2.shape} are incorrect!\n Required X.shape=[N, D] or X2.shape=[B, N, D]")
        return output_shape

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor]=None) -> tf.Tensor:
        """
        X input is P(X), X2 input is Q(X), if X2 not provided X2<-P(X);
        uses self.w weighting (distribution) vector for computing weighted Hellinger kernel
        returns wHK value
            if X == X2 returns tf.Tensor shape [1, N, N]
            if X != X2 returns tf.Tensor shape [1, len(X), 1, len(X2)]
        """
        output_shape = self._handle_k_output_shape(X, X2)
        _X = self.restore(X)
        if X2 is None:
            _X2 = _X
        else:
            _X2 = self.restore(X2)
        assert _X.shape[-1] == self.AA and _X.shape[-2] == self.L, "Input vector X last two dimensions not consistent! (L, AA)"
        assert _X2.shape[-1] == self.AA and _X2.shape[-2] == self.L, "Input vector X2 last two dimensions not consistent! (L, AA)"

        M = self._H(_X, _X2)
        if X2 is None:
            M = tf.linalg.set_diag(M, tf.ones(M.shape[0], dtype=default_float()))
        M = tf.reshape(M, shape=output_shape) # NOTE: external shape check [batch..., N1, batch..., N2] if X!=X2
        return M

    def _get_inner_product(self, X: tf.Tensor, X2: tf.Tensor) -> tf.Tensor:
        """
        Compute RHS of weighted HK equation, as weighting times sqrt(p[a_l,l] x q[a_l,l])
        """
        # M = tf.math.reduce_sum(self.w * tf.sqrt(X[None, ...] * X2[:, None, ...]), axis=-1)
        # NOTE: the einsum and reduce_sum product should be equivalent
        M = tf.einsum('ali,bli->abl', tf.sqrt(self.w*X), tf.sqrt(self.w*X2))
        return tf.math.reduce_prod(M, axis=-1) # product over L, positions factorize

    def _compute_lhs(self, X: tf.Tensor, X2: tf.Tensor) -> tf.Tensor:
        w_p = tf.math.reduce_prod(tf.math.reduce_sum(self.w[None, ...]*X, axis=-1), axis=-1) / 2
        w_q = tf.math.reduce_prod(tf.math.reduce_sum(self.w[None, ...]*X2, axis=-1), axis=-1) / 2
        return w_p[:, None]+tf.transpose(w_q[:, None])
        # return tf.reshape(sum_prod_over_L, shape=(X.shape[0], X2.shape[0]))

    def _H(self, X: tf.Tensor, X2: tf.Tensor) -> tf.Tensor:
        M = self._get_inner_product(X, X2)
        # NOTE: LHS is expectation with equal weight, could have weighting 
        weighted_E = self._compute_lhs(X, X2)
        M = weighted_E - M
        #M = tf.where(M < 0., tf.zeros_like(M), M)
        M = tf.where(M <= 0., tf.zeros_like(M), M) # fix gradients
        M = tf.exp(-tf.sqrt(M) / tf.square(self.lengthscale))
        return M
