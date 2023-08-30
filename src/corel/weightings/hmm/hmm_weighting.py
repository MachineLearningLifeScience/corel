__author__ = 'Simon Bartels'

import numpy as np
import tensorflow as tf
from gpflow import default_float

from corel.util.constants import PADDING_SYMBOL_INDEX
from corel.weightings.abstract_weighting import AbstractWeighting
from corel.weightings.hmm.hmm_forward import forward
from corel.weightings.hmm.load_phmm import load_hmm


class HMMWeighting(AbstractWeighting):
    def __init__(self, hmm, amino_acid_integer_mapping):
        self.s0, self.T, self.em, extra_info_dict = load_hmm(hmm)
        hmm_alphabet = hmm.metadata.alphabet
        self.index_map = {amino_acid_integer_mapping[hmm_alphabet[i]]: i for i in range(len(hmm_alphabet))}

    def expectation(self, p):
        # this implementation currently only supports atomic p

        if not p.dtype.is_integer:
            # bug in tensorflow: 0^0=1. Well, good for me...
            if not tf.reduce_all(tf.pow(p, p) == 1.):
                print(p)
                raise NotImplementedError("Expectations w.r.t. non-atomic distributions not implemented")
            # TODO: get argmax
            p = tf.argmax(p, axis=-1)

        e = np.zeros([p.shape[0], 1])
        for i in range(p.shape[0]):
            # TODO: How to do this in tensorflow?
            #s = p[i, p[i, :] != PADDING_SYMBOL_INDEX]
            p_ = p[i, :].numpy()
            s = p_[p_ != PADDING_SYMBOL_INDEX]
            # if UNKNOWN_AA in s:
            #     p[i] = 0.
            #     continue
            seq_to_int = np.array([self.index_map[s[j]] for j in range(len(s))])
            _, c = forward(self.s0, self.T, self.em, seq_to_int)
            e[i] = np.prod(c)
        return tf.constant(e)
