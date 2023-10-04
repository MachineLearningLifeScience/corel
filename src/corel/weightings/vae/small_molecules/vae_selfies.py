"""This module implements a VAE for SELFIES representations of molecules.

Using the ZINC250k dataset, we train a Variational Autoencoder
that decodes to a categorical distribution.
"""
from typing import List
from typing import Any

import tensorflow as tf
import tensorflow_probability as tfp

from corel.weightings.vae.base import (
    LATENT_DIM,
    DECODER_LAYERS,
    ENCODER_LAYERS,
    DROPOUT_RATE,
    KL_WEIGHT,
    PRIOR_SCALE,
    OFFDIAG,
)
from corel.weightings.vae.base.models import Encoder, Decoder

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class Encoder:
    """A variational encoder that encodes to a Gaussian distribution."""

    def __init__(
        self,
        z_dim: int,
        hidden_dims: List[int],
        input_dims: int,
        n_categories: int,
        prior,
        offdiag=False,
    ) -> None:
        # Size of the latent space
        self.z_dim = z_dim

        # Assuming input_dims == n_categories * sequence_length
        self.input_dims = input_dims

        # Number of categories
        self.n_categories = n_categories

        # Sequence length, taken from the assumption about input_dims
        self.sequence_length = input_dims // n_categories

        # The dense layers
        dense_layers = [
            tfkl.Dense(d, activation=tf.nn.leaky_relu, name=f"enc_layer_{i}")
            for i, d in enumerate(hidden_dims)
        ]

        # The distribution layers
        if offdiag:
            distribution_layers = [
                tfkl.Dense(tfpl.MultivariateNormalTriL.params_size(z_dim)),
                tfpl.MultivariateNormalTriL(
                    z_dim,
                    activity_regularizer=tfpl.KLDivergenceRegularizer(
                        prior, use_exact_kl=True
                    )
                    # , weight=KL_WEIGHT)
                ),
            ]
        else:  # distribution should match with prior
            distribution_layers = [  # NOTE: this can get numerically unstable during inference
                tfkl.Dense(tfpl.IndependentNormal.params_size(z_dim)),
                tfpl.IndependentNormal(
                    event_shape=z_dim,
                    activity_regularizer=tfpl.KLDivergenceRegularizer(
                        prior, use_exact_kl=True
                    )
                    # , weight=KL_WEIGHT)
                ),
            ]

        self.layers = tfk.Sequential(
            [
                tfkl.InputLayer(input_shape=input_dims),
                *dense_layers,
                *distribution_layers,
            ]
        )


class CategoricalDecoder:
    """A decoder that decodes to a categorical distribution."""

    def __init__(
        self,
        z_dim: int,
        hidden_dims: List[int],
        input_dims: int,
        n_categories: int,
        dropout: float = DROPOUT_RATE,
    ) -> None:
        """A decoder that decodes to a categorical distribution.

        Args:
            z_dim (int): The size of the latent space.
            hidden_dims (List[int]): The hidden dimensions of the dense layers.
            input_dims (int): The input dimensions of the decoder
            (i.e. the number of categories times the sequence length).
            n_categories (int): The number of categories.
            dropout (float, optional): The dropout rate. Defaults to DROPOUT_RATE.
        """
        # Size of the latent space
        self.z_dim = z_dim

        # Assuming input_dims == n_categories * sequence_length
        self.input_dims = input_dims

        # Number of categories
        self.n_categories = n_categories
        self.n_classes = n_categories

        # Sequence length, taken from the assumption about input_dims
        self.sequence_length = input_dims // n_categories

        # The dense layers. We assume that the output
        # are the logits of a categorical distribution
        dense_layers = [
            tfkl.Dense(d, activation=tf.nn.leaky_relu, name=f"dec_layer_{i}")
            for i, d in enumerate(hidden_dims)
        ]

        self.layers = tfk.Sequential(
            [
                tfkl.InputLayer(input_shape=[z_dim], name="z_to_dense"),
                # tfkl.Reshape([1, 1, z_dim]),
                *dense_layers,
                tfkl.Dropout(dropout, name="dropout"),
                tfkl.Dense(input_dims, activation=None),
                tfkl.Reshape([self.sequence_length, n_categories]),
                tfpl.DistributionLambda(lambda t: tfd.Categorical(logits=t)),
            ]
        )


class VAESelfies:
    def __init__(
        self,
        z_dim: int,
        input_dims: int,
        n_categories: int,
        encoder_layers=ENCODER_LAYERS,
        decoder_layers=DECODER_LAYERS,
        offdiag=OFFDIAG,
    ) -> None:
        # Defining the hyperparameters
        self.z_dim = z_dim
        self.input_dims = input_dims
        self.n_categories = n_categories

        # Defining the prior
        if offdiag:
            self.prior = tfd.MultivariateNormalTriL(
                loc=tf.zeros(z_dim), scale_tril=tf.eye(z_dim)
            )
        else:
            self.prior = tfd.Independent(
                tfd.Normal(loc=tf.zeros(z_dim), scale=PRIOR_SCALE),
                reinterpreted_batch_ndims=1,
            )

        # Defining the encoder
        self.encoder = Encoder(
            z_dim,
            encoder_layers,
            input_dims=input_dims,
            n_categories=n_categories,
            prior=self.prior,
            offdiag=offdiag,
        )

        # Defining the decoder
        self.decoder = CategoricalDecoder(
            z_dim, decoder_layers, input_dims=input_dims, n_categories=n_categories
        )

        # Putting it together
        self.model = tfk.Model(
            inputs=self.encoder.layers.inputs,
            outputs=self.decoder.layers(self.encoder.layers.outputs[0]),
        )

    @staticmethod
    def neg_ll(x, model):
        return -model.log_prob(x)


if __name__ == "__main__":
    max_sequence_length = 70
    n_categories = 64

    decoder = CategoricalDecoder(
        z_dim=2,
        hidden_dims=[100, 1000],
        input_dims=max_sequence_length * n_categories,
        n_categories=n_categories,
    )

    print(decoder.layers(tf.random.normal([1, 2])))
