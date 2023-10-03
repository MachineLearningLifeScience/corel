import unittest
from itertools import product

import numpy as np

from corel.util.constants import PADDING_SYMBOL_INDEX
from corel.weightings.hmm.hmm_forward import forward, _forward_instable
from corel.weightings.hmm.hmm_weighting import HMMWeighting


class TestHMMweighting(unittest.TestCase):
    def test_expectation(self):
        L = 3
        hmm = TestHMMImplementation()
        p = np.square(np.random.randn(L, hmm.em.shape[1]))
        p /= np.sum(p, axis=-1)[:, None]
        e_ = hmm._expectation(p)
        e = 0.
        for x in product(list(range(hmm.em.shape[1])), repeat=L):
            _, c = forward(hmm.s0, hmm.T, hmm.em, np.array(x))
            e += np.prod(c) * np.prod(p[np.arange(L), x])
        self.assertAlmostEqual(np.log(e), np.log(e_))

    def test_expectation_wrt_atoms(self):
        L = 3
        hmm = TestHMMImplementation()
        x = np.random.randint(0, hmm.em.shape[1], L)
        p = np.zeros([L, hmm.em.shape[1]])
        p[np.arange(L), x] = 1.
        e_ = hmm._expectation(p)
        #_, c_ = _forward_instable(hmm.s0, hmm.T, hmm.em, x)
        _, c = forward(hmm.s0, hmm.T, hmm.em, x)
        log_e = np.sum(np.log(c))
        self.assertAlmostEqual(log_e, np.log(e_))


class TestHMMImplementation(HMMWeighting):
    # noinspection PyMissingConstructor
    def __init__(self):
        S = 5  # state space size
        AA = 11  # number of amino acids
        self.s0 = np.square(np.random.randn(S, 1))
        self.s0 /= np.sum(self.s0)
        self.T = np.square(np.random.randn(S, S))
        self.T = np.diag(1. / np.sum(self.T, axis=1)) @ self.T
        self.em = np.square(np.random.randn(S, AA))
        self.em = np.diag(1. / np.sum(self.em, axis=1)) @ self.em
        hmm_alphabet = [str(i) for i in range(AA)]
        assert(PADDING_SYMBOL_INDEX == 0)
        amino_acid_integer_mapping = {hmm_alphabet[i]: i+1 for i in range(len(hmm_alphabet))}
        self.index_map = {amino_acid_integer_mapping[hmm_alphabet[i]]: i for i in range(len(hmm_alphabet))}


if __name__ == '__main__':
    unittest.main()
