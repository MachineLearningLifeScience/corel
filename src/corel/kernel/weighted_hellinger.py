import gpflow
from gpflow.utilities import positive
import tensorflow as tf
from typing import Optional
import numpy as np

from corel.util.util import handle_batch_shape
from corel.kernel import Hellinger


class WeightedHellinger(Hellinger):
    def __init__(self, z: tf.Tensor, L: int, AA: int, lengthscale: float=1.0, noise: float=0.1, active_dims: Optional[int] = None, name: Optional[str] = None) -> None:
        super().__init__(L=L, AA=AA, active_dims=active_dims, name=name)
        self.z = z
        # TODO assert p in [0,1]
        self.lengthscale = gpflow.Parameter(lengthscale, transform=positive()) # TODO: log transform here?
        self.noise = gpflow.Parameter(noise, transform=positive()) # TODO: check against Kernel Interface

    def K(self, X, X2=None) -> tf.Tensor:
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

    def _H(self, X: tf.Tensor, X2: tf.Tensor):
        raise NotImplementedError("TODO: implement weighting by expected value")
        M = self._get_inner_product(X, X2)
        # TODO: correctly compute the z vector!
        # z = tf.reduce_sum(tf.squeeze(self.z), -1)[None:]
        M = z@tf.transpose(z) - M
        #M[M < 0.] = 0.
        M = tf.where(M < 0., tf.zeros_like(M), M)
        
        M = tf.where(M == 0., tf.zeros_like(M), M) # fix gradients
        M = tf.exp(-tf.sqrt(M) / tf.square(self.lengthscale))
        return M
