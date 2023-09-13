__author__ = 'Simon Bartels'

import tensorflow as tf
from gpflow import default_float

from corel.weightings.abstract_weighting import AbstractWeighting
from corel.weightings.vae.cbas_vae_wrapper import CBASVAEWrapper


class VAEWeighting(AbstractWeighting):
    def __init__(self, AA, L, prefix):
        #raise NotImplementedError("Make sure implementation is correct")
        vae = CBASVAEWrapper(AA=AA, L=L, prefix=prefix)
        # TODO: average over more distributions
        latent_dim = vae.vae.latentDim_
        self.p0 = vae.decode(tf.zeros([1, latent_dim]))

    def expectation(self, p):
        if p.dtype.is_integer:
            #raise NotImplementedError("obtain number of amino acids and make sure that permutation is correct")
            p_ = tf.one_hot(p, 20, dtype=default_float(), axis=-1)
        else:
            p_ = p
        assert(len(p_.shape) == 3)
        return tf.expand_dims(tf.reduce_prod(tf.reduce_sum(self.p0 * p_, axis=-1), axis=-1), -1)
