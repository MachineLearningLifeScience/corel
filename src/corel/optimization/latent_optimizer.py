__author__ = 'Simon Bartels'

import warnings

import numpy as np
import tensorflow as tf
from gpflow import default_float
from trieste.acquisition.optimizer import automatic_optimizer_selector, generate_continuous_optimizer
from trieste.space import SearchSpaceType, TaggedProductSearchSpace, Box
from tensorflow_probability.python.distributions import Categorical

from corel.util.k_best import KeepKBest
from corel.weightings.vae.cbas_vae_wrapper import CBASVAEWrapper


def make_latent_optimizer(batch_size=1, samples_from_proposal=50):
    assert(samples_from_proposal >= batch_size)
    def latent_optimizer(search_space: SearchSpaceType, acquisition_function) -> tf.Tensor:
        raise NotImplementedError("get problem quantities")
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
        bestk = KeepKBest(batch_size, copy=lambda x: x.copy())
        for _ in range(3):
            try:
                z = generate_continuous_optimizer(
                        num_initial_samples=1,
                        num_optimization_runs=1,
                        num_recovery_runs=0
                    )(sp, ac)
                p = vae.decode(z)
                bestk = get_best_of_k(samples_from_proposal, p, acquisition_function, bestk)
                # if v_ > v:
                #     x = x_
                #     v = v_
            except Exception as e:
                warnings.warn("An optimization attempt failed with exception " + str(e))
        selected_seqs, vals = bestk.get()
        return tf.constant(np.concatenate(selected_seqs.tolist()))


def get_best_of_k(k, P, acquisition_function, bestk: KeepKBest) -> KeepKBest:
    seq = tf.argmax(P, axis=-1)
    atom = _seq_to_atom(seq)
    val = acquisition_function(atom)
    bestk.new_val(val, seq.numpy())
    dist = Categorical(P)
    for i in range(k):
        seq_ = dist.sample()
        val_ = acquisition_function(_seq_to_atom(seq_))
        bestk.new_val(val_, seq_.numpy())
    return bestk


def get_best_of_single_site_mutations(P, acquisition_function, bestk: KeepKBest) -> KeepKBest:
    seq = tf.argmax(P, axis=-1)
    atom = _seq_to_atom(seq)
    val = acquisition_function(atom)
    bestk.new_val(val, seq.numpy())

    seq_ = seq.numpy().copy()
    for l in range(P.shape[1]):
        for a in range(1, P.shape[2]):
            seq_[0, l] = a
            val_ = acquisition_function(_seq_to_atom(seq_))
            bestk.new_val(val_, seq_)
        seq_[0, l] = seq[0, l].numpy()
    return bestk


def _seq_to_atom(x):
    return tf.reshape(tf.one_hot(x, depth=20, axis=-1, dtype=default_float()), [x.shape[0], 1, 237*20])
