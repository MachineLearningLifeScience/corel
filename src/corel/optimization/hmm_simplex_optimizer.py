__author__ = 'Simon Bartels'

import time

import numpy as np
import tensorflow as tf
from gpflow import default_float
from poli.core.problem_setup_information import ProblemSetupInformation
from trieste.acquisition.multi_objective import non_dominated
from trieste.space import SearchSpaceType

from corel.aligning_protein_model import AligningProteinModel
from corel.optimization.pareto_frontier_explorer import _padded_until
from corel.optimization.simplex_optimizer import _make_optimizer, _make_initial_distribution_from_sequence
from corel.util.constants import PADDING_SYMBOL_INDEX
from corel.util.k_best import KeepKBest


def make_hmm_simplex_optimizer(problem_info: ProblemSetupInformation, dataset=None, batch_evaluations=1):
    def unaligned_batch_simplex_optimizer(search_space: SearchSpaceType, acquisition_function) -> tf.Tensor:
        assert(isinstance(acquisition_function._model, AligningProteinModel))
        AA = search_space.upper[0].numpy() + 1
        if problem_info.sequences_are_aligned():
            assert(AA == len(problem_info.get_alphabet()))
        else:
            assert(AA - 1 == len(problem_info.get_alphabet()))

        p0 = np.concatenate([np.zeros([acquisition_function._model.distribution.T.shape[0], 1]), acquisition_function._model.distribution.em], axis=1)
        p0 = p0 / np.sum(p0, axis=-1)[:, np.newaxis]  # TODO: should be normalized but isn't. Investigate!
        # TODO: a much better initial distribution is probably doing an encode/decode
        opt = _make_optimizer(p0.shape[0], AA, acquisition_function, p0)
        p = opt()
        # TODO: now we ideally get the viterbi path for all observed sequences and sample according to p
        raise NotImplementedError("still need to decide which sequence to take from the HMM")
        x = tf.expand_dims(tf.concat([tf.argmax(p_) for p_ in p], axis=0), axis=0)
        #print("best acquisition value found: " + str(best_val))
        return x  #tf.concat([x])
    return unaligned_batch_simplex_optimizer
