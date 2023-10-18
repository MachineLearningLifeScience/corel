import unittest
from itertools import product

import numpy as np
import tensorflow as tf

import corel.weightings.hmm.unaligned_hmm_weighting
from corel.util.constants import PADDING_SYMBOL_INDEX
from corel.weightings.hmm.hmm_forward import forward, _forward_instable, tf_forward
from corel.weightings.hmm.unaligned_hmm_weighting import UnalignedHMMWeighting


class TestHMMweighting(unittest.TestCase):
    def test_expectation(self):
        L = 3
        #MAX_LENGTH = L
        corel.weightings.hmm.unaligned_hmm_weighting.MAX_LENGTH = L
        hmm = TestHMMImplementation()
        p = np.square(np.random.randn(*hmm.em.shape))
        p /= np.sum(p, axis=-1)[:, None]
        p = tf.constant(p)
        e_ = hmm._expectation(p)
        print(e_)
        e = tf.zeros(1, dtype=tf.float64)
        for l in range(L):
            for x in product(list(range(hmm.em.shape[1])), repeat=l+1):
                x_ = tf.constant(np.array(x), dtype=tf.int64)
                c = tf_forward(hmm.s0, hmm.T, hmm.em, x_)
                c_ = tf_forward(hmm.s0, hmm.T, p, x_)
                e += tf.reduce_prod(c * c_)
        print(e)
        self.assertAlmostEqual(tf.math.log(e).numpy()[0], tf.math.log(e_).numpy())

    def test_expectation_wrt_atoms(self):
        L = 4
        hmm = TestHMMImplementation()
        x = np.random.randint(0, hmm.em.shape[1], L)
        p = np.zeros([L, hmm.em.shape[1]], dtype=np.int64)
        p[np.arange(L), x] = 1
        p = tf.constant(p)
        e_ = hmm.expectation(p)
        #_, c_ = _forward_instable(hmm.s0, hmm.T, hmm.em, x)
        c = tf_forward(hmm.s0, hmm.T, hmm.em, x)
        log_e = tf.math.log(c)
        self.assertAlmostEqual(log_e, tf.math.log(e_))

    def test_against_naive_implementation(self):
        L = 3
        #MAX_LENGTH = L
        corel.weightings.hmm.unaligned_hmm_weighting.MAX_LENGTH = L
        hmm = TestHMMImplementation()
        p = np.square(np.random.randn(*hmm.em.shape))
        p /= np.sum(p, axis=-1)[:, None]
        p = tf.constant(p)
        S = hmm.T.shape[0]
        E = np.zeros([S, S])
        for s in range(S):
            for t in range(S):
                E[s, t] = tf.reduce_sum(hmm.s0[s] * hmm.em[s, :] * p[t, :] * hmm.s0[t]).numpy()
        e = np.sum(E)
        for l in range(1, L):
            E_ = tf.constant(E.copy())
            for s in range(S):
                for t in range(S):
                    #E[s, t] = tf.reduce_sum(E_ * hmm.T[:, s] * hmm.T[:, t])
                    E[s, t] = tf.transpose(hmm.T[:, s:s+1]) @ E_ @ hmm.T[:, t:t+1] * tf.reduce_sum(hmm.em[s, :] * p[t, :])
            e += tf.reduce_sum(E).numpy()
        print(e)
        e_ = hmm._expectation(p)
        print(e_)
        self.assertAlmostEqual(np.log(e), tf.math.log(e_).numpy())



class TestHMMImplementation(UnalignedHMMWeighting):
    # noinspection PyMissingConstructor
    def __init__(self):
        S = 5  # state space size
        AA = 11  # number of amino acids
        s0 = np.square(np.random.randn(S, 1))
        s0 /= np.sum(s0)
        self.s0 = tf.constant(s0)
        T = np.square(np.random.randn(S, S))
        T = np.diag(1. / np.sum(T, axis=1)) @ T
        self.T = tf.constant(T)
        em = np.square(np.random.randn(S, AA))
        em = np.diag(1. / np.sum(em, axis=1)) @ em
        self.em = tf.constant(em)
        hmm_alphabet = [str(i) for i in range(AA)]
        assert(PADDING_SYMBOL_INDEX == 0)
        amino_acid_integer_mapping = {hmm_alphabet[i]: i for i in range(len(hmm_alphabet))}
        self.index_map = {amino_acid_integer_mapping[hmm_alphabet[i]]: i for i in range(len(hmm_alphabet))}
        self.index_permutation = np.array(list(amino_acid_integer_mapping[hmm_alphabet[i]] for i in range(len(hmm_alphabet))))


if __name__ == '__main__':
    unittest.main()
