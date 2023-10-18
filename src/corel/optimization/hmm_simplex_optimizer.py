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
from corel.weightings.hmm.hmm_viterbi import viterbi
from corel.weightings.hmm.unaligned_hmm_weighting import UnalignedHMMWeighting


def make_hmm_simplex_optimizer(problem_info: ProblemSetupInformation, dataset=None, batch_evaluations=1):
    if dataset is None:
        # observations_handle = lambda ac: tf.concat([ac._model._models[i].dataset.observations for i in range(len(ac._model._models))], axis=-1)
        # inputs_handle = lambda ac: ac._model._models[0].dataset.query_points
        observations_handle = lambda ac: ac._model.dataset.observations
        inputs_handle = lambda ac: ac._model.dataset.query_points

    def unaligned_batch_simplex_optimizer(search_space: SearchSpaceType, acquisition_function) -> tf.Tensor:
        assert(isinstance(acquisition_function._model, AligningProteinModel))
        hmm: UnalignedHMMWeighting = acquisition_function._model.distribution
        AA = search_space.upper[0].numpy() + 1
        if problem_info.sequences_are_aligned():
            assert(AA == len(problem_info.get_alphabet()))
        else:
            assert(AA - 1 == len(problem_info.get_alphabet()))


        p0 = np.concatenate([np.zeros([hmm.T.shape[0], 1]), acquisition_function._model.distribution.em], axis=1)
        p0 = p0 / np.sum(p0, axis=-1)[:, np.newaxis]  # TODO: should be normalized but isn't. Investigate!
        # TODO: a much better initial distribution is probably doing an encode/decode
        opt = _make_optimizer(p0.shape[0], AA, acquisition_function, p0)
        p = opt()
        # TODO: now we ideally get the viterbi path for all observed sequences and sample according to p
        state_path = viterbi(inputs_handle(acquisition_function)[0].numpy(), hmm.T, hmm.em, hmm.s0)
        x = tf.expand_dims(tf.concat([tf.argmax(p[s]) for s in state_path], axis=0), axis=0)
        #print("best acquisition value found: " + str(best_val))
        return x  #tf.concat([x])
    return unaligned_batch_simplex_optimizer
