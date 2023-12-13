"""This module implements a VAE for SELFIES representations of molecules.

Using the ZINC250k dataset, we train a Variational Autoencoder
that decodes to a categorical distribution.
"""
from typing import List, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow import default_float

from corel.weightings.vae.base import (
    LATENT_DIM,
    DECODER_LAYERS,
    ENCODER_LAYERS,
    DROPOUT_RATE,
    KL_WEIGHT,
    PRIOR_SCALE,
    OFFDIAG,
)
from corel.weightings.abstract_weighting import AbstractWeighting

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class EncoderSelfies:
    """A variational encoder that encodes to a Gaussian distribution."""

    def __init__(
        self,
        z_dim: int,
        hidden_dims: List[int],
        input_dims: Tuple[int, int],
        n_categories: int,
        prior,
        offdiag=False,
    ) -> None:
        # Size of the latent space
        self.z_dim = z_dim

        # Assuming input_dims = [sequence_length, n_categories]
        self.input_dims = input_dims

        # Number of categories
        self.n_categories = n_categories

        # Sequence length, taken from the assumption about input_dims
        self.sequence_length = self.input_dims[0]

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
                        prior,
                        use_exact_kl=True,
                        weight=0.01,
                    ),
                ),
            ]
        else:  # distribution should match with prior
            distribution_layers = (
                [  # NOTE: this can get numerically unstable during inference
                    tfkl.Dense(tfpl.IndependentNormal.params_size(z_dim)),
                    tfpl.IndependentNormal(
                        event_shape=z_dim,
                        activity_regularizer=tfpl.KLDivergenceRegularizer(
                            prior,
                            use_exact_kl=True,
                            weight=0.01,
                        ),
                    ),
                ]
            )

        self.layers = tfk.Sequential(
            [
                tfkl.InputLayer(input_shape=input_dims),
                tfkl.Flatten(),
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
        input_dims: Tuple[int, int],
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

        # Assuming input_dims = [sequence_length, n_categories]
        self.input_dims = input_dims

        # Number of categories
        self.n_categories = n_categories
        self.n_classes = n_categories

        # Sequence length, taken from the assumption about input_dims
        self.sequence_length = self.input_dims[0]

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
                # tfkl.Dropout(dropout, name="dropout"),
                tfkl.Dense(self.sequence_length * self.n_categories, activation=None),
                tfkl.Reshape([self.sequence_length, self.n_categories]),
                tfpl.OneHotCategorical(event_size=self.sequence_length),
            ]
        )


class VAESelfies(AbstractWeighting):
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
        self.encoder = EncoderSelfies(
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

    def expectation(self, p: tf.Tensor) -> tf.Tensor:
        """
        Computes E_{p(x|z=0)}[p], which is the weighting
        of a given probability distribution used in the
        Hellinger kernel.

        (Taken from the vae weighting class in the CBAS folder)
        """
        if p.dtype.is_integer:
            raise NotImplementedError(
                "obtain number of amino acids and make sure that permutation "
                "is correct"
            )
            p_ = tf.one_hot(p, 20, dtype=default_float(), axis=-1)
        else:
            p_ = p
        assert len(p_.shape) == 3

        return tf.expand_dims(
            tf.reduce_prod(tf.reduce_sum(self.p0 * p_, axis=-1), axis=-1), -1
        )

    def __call__(self, x: tf.Tensor) -> tf.Tensor:
        return self.expectation(x)


if __name__ == "__main__":
    # When run, this script tests whether the information
    # is passed correctly through the encoder and decoder.
    from corel.util.small_molecules.data import load_zinc_250k_dataset

    # Defining hyperparameters
    max_sequence_length = 70
    n_categories = 64
    offdiag = OFFDIAG
    z_dim = LATENT_DIM

    # Defining the prior
    if offdiag:
        prior = tfd.MultivariateNormalTriL(
            loc=tf.zeros(z_dim), scale_tril=tf.eye(z_dim)
        )
    else:
        prior = tfd.Independent(
            tfd.Normal(
                loc=tf.zeros(z_dim), scale=PRIOR_SCALE
            ),  # TODO: LaPlace prior instead of Normal prior?
            reinterpreted_batch_ndims=1,
        )

    # Testing the encoder
    encoder = EncoderSelfies(
        z_dim=2,
        hidden_dims=[100, 1000],
        input_dims=(max_sequence_length, n_categories),
        n_categories=n_categories,
        prior=prior,
    )

    # Loading up an example input
    all_onehot_arrays = load_zinc_250k_dataset()
    dataset_size, sequence_length, n_categories = all_onehot_arrays.shape
    all_onehot_arrays = tf.constant(
        all_onehot_arrays,
        dtype=tf.float32,
        name="all_onehot_arrays",
    )
    x0 = all_onehot_arrays[:1]

    # Passing the input through the encoder
    latent_dist = encoder.layers(x0)
    one_latent_sample = latent_dist.sample()
    print("Example sample from the latent dist q(z|x): ", one_latent_sample)

    decoder = CategoricalDecoder(
        z_dim=2,
        hidden_dims=[100, 1000],
        input_dims=(max_sequence_length, n_categories),
        n_categories=n_categories,
    )

    dist_ = decoder.layers(one_latent_sample)
    print("Example sample from the conditional dist p(x|z): ", dist_.sample())
