from typing import List
from typing import Any

import tensorflow as tf
# import tensorflow.compat.v2 as tf
# tf.enable_v2_behavior()

import tensorflow_datasets as tfds
import tensorflow_probability as tfp

from corel.weightings.vae.base import LATENT_DIM, DECODER_LAYERS, ENCODER_LAYERS, DROPOUT_RATE, KL_WEIGHT, PRIOR_SCALE, OFFDIAG

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class Encoder:
    def __init__(self, z_dim: int, hidden_dims: List[int], input_dims: int, n_categories: int, prior, offdiag=False) -> None:
        self.z_dim = z_dim
        self.input_dims = input_dims
        self.n_classes = n_categories
        dense_layers = [tfkl.Dense(d, activation=tf.nn.leaky_relu, name=f"enc_layer_{i}") for i, d in enumerate(hidden_dims)] 
        if offdiag:
            distribution_layers = [
                tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(z_dim)),
                tfpl.MultivariateNormalTriL(
                    z_dim,
                    activity_regularizer=tfpl.KLDivergenceRegularizer(prior, use_exact_kl=True)
                    #, weight=KL_WEIGHT)
                )]
        else: # distribution should match with prior
            distribution_layers = [ # NOTE: this can get numerically unstable during inference
                tfkl.Dense(tfpl.IndependentNormal.params_size(z_dim)),
                tfpl.IndependentNormal(
                    event_shape=z_dim,
                    activity_regularizer=tfpl.KLDivergenceRegularizer(prior, use_exact_kl=True)
                    #, weight=KL_WEIGHT)
                )]
        self.layers = tfk.Sequential([
            tfkl.InputLayer(input_shape=input_dims),
            tfkl.Flatten(),
            *dense_layers,
            *distribution_layers
        ])


class Decoder:
    def __init__(self, z_dim: int, hidden_dims: List[int], input_dims: int, n_categories: int, dropout: float=DROPOUT_RATE) -> None:
        self.z_dim = z_dim
        self.input_dims = input_dims
        self.n_classes = n_categories
        dense_layers = [tfkl.Dense(d, activation=tf.nn.leaky_relu, name=f"dec_layer_{i}") for i,d in enumerate(hidden_dims[:-1])]
        dense_layers += [tfkl.Dense(hidden_dims[-1], activation=tf.nn.softmax)]
        self.layers = tfk.Sequential([
            tfkl.InputLayer(input_shape=[z_dim], name="z_to_dense"),
            tfkl.Reshape([1, 1, z_dim]),
            *dense_layers,
            tfkl.Dropout(dropout, name="dropout"),
            tfkl.Dense(tfpl.IndependentBernoulli.params_size(input_dims)), # no activation layer into Bernoulli
            tfpl.IndependentBernoulli(input_dims, tfd.Bernoulli.logits),
        ])


class VAE:
    def __init__(self, z_dim: int, input_dims: int, n_categories: int, encoder_layers = ENCODER_LAYERS, decoder_layers = DECODER_LAYERS, offdiag=OFFDIAG) -> None:
        if offdiag:
            self.prior = tfd.MultivariateNormalTriL(
                loc=tf.zeros(z_dim), scale_tril=tf.eye(z_dim)
            )
        else:
            self.prior = tfd.Independent(
                tfd.Normal(loc=tf.zeros(z_dim), scale=PRIOR_SCALE), # TODO: LaPlace prior instead of Normal prior?
                reinterpreted_batch_ndims=1,  
            )
        self.encoder = Encoder(z_dim, encoder_layers, input_dims=input_dims, n_categories=n_categories, prior=self.prior, offdiag=offdiag)
        self.decoder = Decoder(z_dim, decoder_layers, input_dims=input_dims, n_categories=n_categories)
        self.model = tfk.Model(inputs=self.encoder.layers.inputs, outputs=self.decoder.layers(self.encoder.layers.outputs[0]))
    
    @staticmethod
    def neg_ll(x, model):
        return -model.log_prob(x)

    def p(self, x: tf.Tensor, dtype=tf.float64) -> tf.Tensor:
        logits_z0 = self.decoder.layers(x)
        logits_z0 = tf.cast(logits_z0, dtype)
        ps = tf.sigmoid(logits_z0) # TFP: "the probability of an [Bernoulli] event is sigmoid(logits)"
        ps = tf.constant(ps.numpy() / tf.reduce_sum(ps, axis=-1).numpy()[..., None]) # probits need to sum to one
        return ps