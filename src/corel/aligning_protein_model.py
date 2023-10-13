__author__ = 'Simon Bartels'

import numpy as np
import tensorflow as tf
from trieste.data import Dataset

from corel.protein_model import ProteinModel
from corel.util.constants import PADDING_SYMBOL_INDEX
from corel.weightings.hmm.hmm_forward import forward, tf_forward
from corel.weightings.hmm.unaligned_hmm_weighting import UnalignedHMMWeighting


class AligningProteinModel(ProteinModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (isinstance(self.distribution, UnalignedHMMWeighting))

    def _evaluate_data_sequences_in_query_distribution(self, p_query):
        #qs = np.prod(p_query.numpy()[0, np.arange(p_query.shape[1]), self.dataset.query_points.numpy()], axis=-1)
        #qs = tf.constant(qs.reshape([p_query.shape[0], self.dataset.query_points.shape[0]]))
        N = self.dataset.query_points.shape[0]
        qs = np.zeros([p_query.shape[0], N])
        assert (p_query.shape[0] == 1)
        def _get_proba_of_data_sequence(i):
            p_ = self.dataset.query_points[i, :].numpy()
            s = p_[p_ != PADDING_SYMBOL_INDEX]
            # if UNKNOWN_AA in s:
            #     p[i] = 0.
            #     continue
            seq_to_int = np.array([self.distribution.index_map[s[j]] for j in range(len(s))])
            assert (PADDING_SYMBOL_INDEX == 0)
            offset = 1  # TODO: check if problem is aligned or not (but it wouldn't make much sense to use this model if it was)
            return tf_forward(tf.constant(self.distribution.s0), tf.constant(self.distribution.T), p_query[0, :, offset:], seq_to_int)
        return tf.expand_dims(tf.stack([_get_proba_of_data_sequence(i) for i in range(N)], 0), 0)
