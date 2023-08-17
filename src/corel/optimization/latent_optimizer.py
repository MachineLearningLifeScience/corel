__author__ = 'Simon Bartels'

import warnings

import numpy as np
import tensorflow as tf
from gpflow import default_float
from trieste.acquisition.optimizer import automatic_optimizer_selector, generate_continuous_optimizer
from trieste.space import SearchSpaceType, TaggedProductSearchSpace, Box
from tensorflow_probability.python.distributions import Categorical

from corel.weightings.vae.cbas_vae_wrapper import CBASVAEWrapper


def latent_optimizer(search_space: SearchSpaceType, acquisition_function) -> tf.Tensor:
    vae = CBASVAEWrapper(AA=20, L=237)

    def make_ac():
        def ac(z):
            assert(len(z.shape) == 3)
            assert(z.shape[1] == 1)
            z = tf.reshape(z, [z.shape[0], z.shape[-1]])
            return acquisition_function(tf.reshape(vae.decode(z), [z.shape[0], 1, 237*20]))
        return ac
    ac = make_ac()

    #sp = TaggedProductSearchSpace(20 * [Box(lower=[-3.], upper=[3.])])
    sp = Box(lower=-3 * tf.ones(20), upper=3 * tf.ones(20))
    #p = automatic_optimizer_selector(sp, ac)
    x, v = None, -np.infty
    for _ in range(3):
        try:
            z = generate_continuous_optimizer(
                    num_initial_samples=1,
                    num_optimization_runs=1,
                    num_recovery_runs=0
                )(sp, ac)
            p = vae.decode(z)
            x_, v_ = get_best_of_k(50, p, acquisition_function)
            if v_ > v:
                x = x_
                v = v_
        except Exception as e:
            warnings.warn("An optimization attempt failed with exception " + str(e))
    x_, v_ = get_best_of_single_site_mutations(p, acquisition_function)
    if v_ > v:
        x = x_

    #x = tf.argmax(p, axis=-1)
    # TODO: REMOVE
    return tf.cast(x, tf.int32)


def get_best_of_k(k, P, acquisition_function):
    seq = tf.argmax(P, axis=-1)
    atom = _seq_to_atom(seq)
    val = acquisition_function(atom)
    dist = Categorical(P)
    for i in range(k):
        seq_ = dist.sample()
        val_ = acquisition_function(_seq_to_atom(seq_))
        # it appears that Trieste is MAXIMIZING acquisition functions
        if val_ > val:
            seq = seq_
            val = val_
    return seq, val


def get_best_of_single_site_mutations(P, acquisition_function):
    seq = tf.argmax(P, axis=-1)
    atom = _seq_to_atom(seq)
    val = acquisition_function(atom)

    seq_ = seq.numpy().copy()
    for l in range(P.shape[1]):
        for a in range(1, P.shape[2]):
            seq_[0, l] = a
            val_ = acquisition_function(_seq_to_atom(seq_))
            if val_ > val:
                seq = tf.constant(seq_.copy())
                val = val_
        seq_[0, l] = seq[0, l].numpy()
    return seq, val


def _seq_to_atom(x):
    return tf.reshape(tf.one_hot(x, depth=20, axis=-1, dtype=default_float()), [x.shape[0], 1, 237*20])
