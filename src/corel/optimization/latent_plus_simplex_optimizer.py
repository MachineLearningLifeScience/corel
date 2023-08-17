__author__ = 'Simon Bartels'

import warnings

import numpy as np
import tensorflow as tf
from gpflow import default_float
from trieste.acquisition.optimizer import automatic_optimizer_selector, generate_continuous_optimizer
from trieste.space import SearchSpaceType, TaggedProductSearchSpace, Box

from corel.optimization.latent_optimizer import get_best_of_k, get_best_of_single_site_mutations
from corel.weightings.vae.cbas_vae_wrapper import CBASVAEWrapper
from simplex_optimizer import _make_optimizer as _make_simplex_optimizer


def combined_optimizer(search_space: SearchSpaceType, acquisition_function) -> tf.Tensor:
    raise NotImplementedError("get problem quantities")
    L = 237
    AA = 20
    vae = CBASVAEWrapper(AA=AA, L=L)

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
    x, p, v = None, None, -np.infty
    for _ in range(3):
        try:
            z = generate_continuous_optimizer(
                    num_initial_samples=1,
                    num_optimization_runs=1,
                    num_recovery_runs=0
                )(sp, ac)
        except Exception as e:
            warnings.warn("An optimization attempt failed with exception " + str(e))
            continue
        p_ = vae.decode(z)
        p_ = _make_simplex_optimizer(search_space, acquisition_function, p_.numpy()[0, ...])()
        p_ = tf.expand_dims(tf.concat([tf.transpose(p_[i].value()) for i in range(L)], axis=0), axis=0)
        x_, v_ = get_best_of_k(50, p_, acquisition_function)
        if v_ > v:
            x = x_
            p = p_
            v = v_
    x_, v_ = get_best_of_single_site_mutations(p, acquisition_function)
    if v_ > v:
        x = x_

    #x = tf.argmax(p, axis=-1)
    # TODO: REMOVE
    return tf.cast(x, tf.int32)
