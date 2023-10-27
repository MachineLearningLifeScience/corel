import unittest

import numpy as np

from corel.weightings.hmm.hmm_viterbi import viterbi


class TestViterbi(unittest.TestCase):
    def test_viterbi(self):
        S = 5
        AA = 9
        L = 7
        s0 = np.square(np.random.randn(S, 1))
        s0 /= np.sum(s0)
        T = np.square(np.random.randn(S, S))
        T = np.diag(1. / np.sum(T, axis=1)) @ T
        em = np.square(np.random.randn(S, AA))
        em = np.diag(1. / np.sum(em, axis=1)) @ em

        x = np.zeros(L, dtype=np.int64)
        y = np.zeros_like(x)
        x[0] = np.argmax(s0)
        y[0] = np.argmax(em[x[0], :])
        for j in range(1, L):
            x[j] = np.argmax(T[x[j-1], :])
            y[j] = np.argmax(em[x[j], :])
        x_ = viterbi(y, T, em, np.squeeze(s0))
        np.testing.assert_equal(x, x_)


if __name__ == '__main__':
    unittest.main()
