# TODO: test expectection by iterating over VAE sequence
import pytest
import tensorflow as tf
import numpy as np

import sys
if "/Users/rcml/corel/" not in sys.path: # TODO: refactor
    sys.path.append("/Users/rcml/corel/")
from experiments.assets.data.rfp_fam import rfp_train_dataset
from corel.weightings.vae.base.models import VAE
from corel.weightings.vae.base.train_vae import _preprocess
from corel.weightings.vae.base.vae_factory import VAEFactory

train_dataset = rfp_train_dataset.map(_preprocess).batch(128).prefetch(tf.data.AUTOTUNE).shuffle(123)
x0 = next(iter(train_dataset))[0][0]

vae_dict = dict( # TODO: refactor this as part of a central config
    z_dim=2, 
    input_dims=x0.shape,
    n_categories=tf.constant(20), # NOTE: see tf.constant(len(AMINO_ACIDS))
    encoder_layers=[1000, 250], 
    decoder_layers=[250, 1000],
)

def test_vae_likelihoods_some_to_one():
    # TODO: load all VAEs
    # across L all category likelihoods should be close to 1 if tf.reduce_sum(-1)
    z = tf.zeros((1,2))
    vae = VAEFactory("results/models/vae_z_2_rfp_fam.ckpt", None, **vae_dict).create(None)
    ps = vae.p(z)
    summed_ps = tf.reduce_sum(ps, -1).numpy()
    np.testing.assert_array_almost_equal(summed_ps, np.ones(shape=summed_ps.shape), decimal=10e-7)