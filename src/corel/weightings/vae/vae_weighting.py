__author__ = 'Simon Bartels'

import tensorflow as tf
from gpflow import default_float

from corel.weightings.vae.cbas_vae_wrapper import CBASVAEWrapper


class VAEWeighting:
    def __init__(self):
        raise NotImplementedError("Make sure implementation is correct")
        vae = CBASVAEWrapper(AA=20, L=237)
        # TODO: average over more distributions
        self.p0 = vae.decode(tf.zeros([1, 20]))

    def __call__(self, *args, **kwargs):
        return self.expectation(*args, **kwargs)

    def expectation(self, p):
        if p.dtype.is_integer:
            p_ = tf.one_hot(p, 20, dtype=default_float(), axis=-1)
        else:
            p_ = p
        assert(len(p_.shape) == 3)
        return tf.expand_dims(tf.reduce_prod(tf.reduce_sum(self.p0 * p_, axis=-1), axis=-1), -1)
