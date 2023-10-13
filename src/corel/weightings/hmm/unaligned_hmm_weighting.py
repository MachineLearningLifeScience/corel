__author__ = 'Simon Bartels'

import numpy as np
import tensorflow as tf
from gpflow import default_float

from corel.util.constants import PADDING_SYMBOL_INDEX
from corel.weightings.abstract_weighting import AbstractWeighting
from corel.weightings.hmm.hmm_forward import forward
from corel.weightings.hmm.hmm_weighting import HMMWeighting
from corel.weightings.hmm.load_phmm import load_hmm


class UnalignedHMMWeighting(HMMWeighting):
    def expectation(self, p):
        """
        This function assumes that p is either an atom OR that p is an emission probability for the same HMM that we use
         for weighing.
        ATTENTION: p is NOT a product distribution of length L!!!
        :param p:
        :return:
        """
        return super().expectation(p)

    def _expectation(self, p):
        """
        This function assumes that p is either an atom OR that p is an emission probability for the same HMM that we use
         for weighing.
        ATTENTION: p is NOT a product distribution of length L!!!
        :param p:
        :return:
        """
        # TODO: is this correct? test against implementation in hmm
        perm_matrix = np.eye(p.shape[1])
        perm_matrix = perm_matrix[:, self.index_permutation]

        self.max_length = 4  # TODO: read out from problem
        #assert all(p.shape[i] == self.em.shape[i] for i in range(len(self.em.shape)))
        #E = tf.zeros([self.T.shape[0], self.T.shape[0]])
        # TODO: check if this matrix multiplication is efficient! It's not! It's an outer product!!!
        PQ = self.em @ tf.transpose(p @ perm_matrix)
        E = tf.linalg.diag(self.s0) @ PQ @ tf.linalg.diag(self.s0)
        e = tf.zeros(1, dtype=tf.float64)
        for l in range(1, self.max_length):
            e = e + tf.reduce_sum(E)
            # TODO: is it necessary to post multiply with transition to the termination state?
            E = self.T @ (PQ * E) @ tf.transpose(self.T)
        return e
