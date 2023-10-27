__author__ = 'Simon Bartels'

import numpy as np
import tensorflow as tf
from gpflow import default_float

from corel.util.constants import PADDING_SYMBOL_INDEX
from corel.weightings.abstract_weighting import AbstractWeighting
from corel.weightings.hmm.hmm_forward import forward
from corel.weightings.hmm.load_phmm import load_hmm


class HMMWeighting(AbstractWeighting):
    def __init__(self, s0, T, em, hmm_alphabet, amino_acid_integer_mapping):
        assert(len(s0.shape) == 2)
        S = s0.shape[0]
        assert(s0.shape[1] == 1)
        np.testing.assert_almost_equal(np.sum(s0), 1.)
        self.s0 = s0
        assert(T.shape[0] == S == T.shape[1])
        np.testing.assert_almost_equal(np.sum(T, axis=-1), np.ones(S))
        self.T = T
        assert(em.shape[0] == S)
        np.testing.assert_almost_equal(np.sum(em, axis=-1), np.ones(S))
        self.em = em
        self.index_map = {amino_acid_integer_mapping[hmm_alphabet[i]]: i for i in range(len(hmm_alphabet))}
        #self.index_permutation = [self.index_map[i] for i in np.sort(list(self.index_map.keys()))]
        self.index_permutation = np.array(list(amino_acid_integer_mapping[hmm_alphabet[i]] for i in range(len(hmm_alphabet))))

    def expectation(self, p):
        p_is_atomic = True

        if not p.dtype.is_integer:
            # bug in tensorflow: 0^0=1. Well, good for me...
            #p_is_atomic = tf.reduce_all(tf.pow(p, p) == 1.)
            # the line below does not exploit a bug
            p_is_atomic = tf.reduce_all(tf.square(p) == p)
            assert(len(p.shape) == 3)

            if p_is_atomic:
                p = tf.argmax(p, axis=-1)

        e = np.zeros([p.shape[0], 1])
        for i in range(p.shape[0]):
            if p_is_atomic:
                # TODO: How to do this in tensorflow?
                #s = p[i, p[i, :] != PADDING_SYMBOL_INDEX]
                p_ = p[i, :].numpy()
                s = p_[p_ != PADDING_SYMBOL_INDEX]
                # if UNKNOWN_AA in s:
                #     p[i] = 0.
                #     continue
                seq_to_int = np.array([self.index_map[s[j]] for j in range(len(s))])
                _, c = forward(self.s0, self.T, self.em, seq_to_int)
                # TODO: it might be necessary to post-multiply transitioning into the last state!
                # otherwise short sequences have much higher probability than they should!
                e[i] = np.prod(c)
            else:
                # The permutation should take care of the padding symbol.
                # Particularly, if the problem is unaligned: p.shape[1] - 1 == self.em.shape[1]
                # assert p.shape[1] == self.em.shape[1], \
                #     (f"Input distribution is over {p.shape[1]} elements whereas the HMM is over {self.em.shape[1]}. "
                #      f"Did you maybe forget to take care of a padding symbol?")
                #p_ = p.numpy()[i, :, self.index_permutation].transpose()  # for some reason numpy swaps the dimensions with this operation!
                p_ = p[i, :, :]
                e[i] = self._expectation(p_)
        return tf.constant(e)

    def _expectation(self, p_):
        p = p_.numpy()[:, self.index_permutation]  #.transpose()  # for some reason numpy swaps the dimensions with this operation!
        return self._expectation_ref(p)
        # TODO:  implement more efficiently
        # assert p.shape[1] == self.em.shape[1], \
        #     (f"Input distribution is over {p.shape[1]} elements whereas the HMM is over {self.em.shape[1]}. "
        #      f"Did you maybe forget to take care of a padding symbol?")
        temp_old = np.ones([self.T.shape[0], 1])
        temp_new = np.zeros_like(temp_old)
        for l in range(p.shape[0] - 1, 0, -1):
            for s_ in range(self.T.shape[0]):
                #for s in range(self.T.shape[0]):
                temp_new = np.sum(p[l, :] * self.em[s, :]) * self.T[s_, :] @ temp_old
            temp_old[:] = temp_new[:]
            temp_new[:] = 0.

        for s in range(self.T.shape[0]):
            temp_new[s] += np.sum(p[0, :] * self.em[s, :]) * temp_old[s]

        return np.sum(self.s0.flatten() * temp_new)

    def _expectation_ref(self, p: np.ndarray):
        # assert p.shape[1] == self.em.shape[1], \
        #     (f"Input distribution is over {p.shape[1]} elements whereas the HMM is over {self.em.shape[1]}. "
        #      f"Did you maybe forget to take care of a padding symbol?")
        temp_old = np.ones(self.T.shape[0])
        temp_new = np.zeros_like(temp_old)
        for l in range(p.shape[0] - 1, 0, -1):
            for s_ in range(self.T.shape[0]):
                for s in range(self.T.shape[0]):
                    temp_new[s_] += np.sum(p[l, :] * self.em[s, :]) * self.T[s_, s] * temp_old[s]
            temp_old[:] = temp_new[:]
            temp_new[:] = 0.

        for s in range(self.T.shape[0]):
            temp_new[s] += np.sum(p[0, :] * self.em[s, :]) * temp_old[s]

        return np.sum(self.s0.flatten() * temp_new)
