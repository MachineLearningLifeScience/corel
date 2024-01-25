__author__ = 'Simon Bartels'

import tensorflow as tf
from gpflow import default_float

from corel.weightings.abstract_weighting import AbstractWeighting
from corel.weightings.vae.cbas.cbas_vae_wrapper import CBASVAEWrapper


class VAEWeighting(AbstractWeighting):
    def __init__(self, vae: CBASVAEWrapper):
        # TODO: average over more distributions
        latent_dim = vae.vae.latentDim_
        self.vae = vae
        self.p0 = vae.decode(tf.zeros([1, latent_dim]))

    def expectation(self, p):
        if p.dtype.is_integer:
            #raise NotImplementedError("obtain number of amino acids and make sure that permutation is correct")
            p_ = tf.one_hot(p, 20, dtype=default_float(), axis=-1)
        else:
            p_ = p
        assert(len(p_.shape) == 3)
        return tf.expand_dims(tf.reduce_prod(tf.reduce_sum(self.p0 * p_, axis=-1), axis=-1), -1)

    def __call__(self, *args, **kwargs):
        return self.expectation(*args, **kwargs)

    def get_training_data(self):
        if "cbas" in self.vae.__class__.__name__.lower():
            return self.vae.get_training_data()
        else:
            raise NotImplementedError("Training data loading only implemented for CBas weighting!")
    
    def get_training_labels(self):
        if "cbas" in self.vae.__class__.__name__.lower():
            return self.vae.get_training_labels()
        else:
            raise NotImplementedError("Training data loading only implemented for CBas weighting!")

