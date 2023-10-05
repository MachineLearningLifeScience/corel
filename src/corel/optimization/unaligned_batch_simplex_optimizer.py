__author__ = 'Simon Bartels'

import time

import numpy as np
import tensorflow as tf
from gpflow import default_float
from poli.core.problem_setup_information import ProblemSetupInformation
from trieste.acquisition.multi_objective import non_dominated
from trieste.space import SearchSpaceType

from corel.optimization.pareto_frontier_explorer import _padded_until
from corel.optimization.simplex_optimizer import _make_optimizer, _make_initial_distribution_from_sequence
from corel.util.constants import PADDING_SYMBOL_INDEX
from corel.util.k_best import KeepKBest


def make_unaligned_batch_simplex_optimizer(problem_info: ProblemSetupInformation, dataset=None, batch_evaluations=1):
    if dataset is None:
        # observations_handle = lambda ac: tf.concat([ac._model._models[i].dataset.observations for i in range(len(ac._model._models))], axis=-1)
        # inputs_handle = lambda ac: ac._model._models[0].dataset.query_points
        observations_handle = lambda ac: ac._model.dataset.observations
        inputs_handle = lambda ac: ac._model.dataset.query_points
    else:
        # TODO: THIS DOES NOT WORK!
        raise NotImplementedError("this does not work! I need the most recent data!")
        observations_handle = lambda ac: dataset

    def unaligned_batch_simplex_optimizer(search_space: SearchSpaceType, acquisition_function) -> tf.Tensor:
        #AA = len(problem_info.get_alphabet())
        AA = search_space.upper[0].numpy() + 1
        if problem_info.sequences_are_aligned():
            assert(AA == len(problem_info.get_alphabet()))
        else:
            assert(AA - 1 == len(problem_info.get_alphabet()))

        # padding symbol index is taken care of in HMM
        # if not problem_info.sequences_are_aligned():
        #     assert(PADDING_SYMBOL_INDEX == 0)
        #     AA = AA - 1
        _, idx = non_dominated(observations_handle(acquisition_function))
        proposal_seqs = inputs_handle(acquisition_function)[idx]
        selected_seqs = []
        for _round, i in enumerate(np.random.randint(0, proposal_seqs.shape[0], batch_evaluations)):
            print("mutating candidate " + str(_round+1) + " of " + str(batch_evaluations))
            t = time.time()
            seq = proposal_seqs[i:i+1, ...]
            seq = seq[:, :_padded_until(seq)].numpy().squeeze()
            raise NotImplementedError("We are getting into shenanigans here.")
            p0 = _make_initial_distribution_from_sequence(AA, seq, problem_info.sequences_are_aligned())
            # TODO: a much better initial distribution is probably doing an encode/decode
            opt = _make_optimizer(len(seq), AA, acquisition_function, p0)
            p = opt()
            x = tf.expand_dims(tf.concat([tf.argmax(p_) for p_ in p], axis=0), axis=0)
            selected_seqs.append(x)
            t = time.time() - t
            print("time: " + str(t))
        #print("best acquisition value found: " + str(best_val))
        return tf.concat(selected_seqs)
    return unaligned_batch_simplex_optimizer


def _make_initial_distribution_from_sequence(AA: int, seq: np.ndarray, aligned: bool) -> np.ndarray:
    """

    :param AA:
        number of amino acids (including padding symbol in the unaligned case)
    :param seq:
    :param aligned:
    :return:
    """
    assert(len(seq.shape) == 1)
    L = seq.shape[0]
    # x0 = tf.expand_dims(search_space.lower, axis=0)
    #normalize = lambda x: x / tf.reduce_sum(x)
    # xs = [tf.Variable(normalize(tf.random.uniform([AA, 1], dtype=default_float()))) for _ in range(L)]
    # xs = [tf.Variable(normalize(tf.ones([AA, 1], dtype=default_float()))) for _ in range(L)]
    epsilon = 1e-8
    x0 = epsilon * np.ones([L, AA])
    if aligned:
        one_minus_Z = 1. - (AA - 1) * epsilon
        x0[np.arange(L), seq] = one_minus_Z
    else:
        one_minus_Z = 1. - (AA - 2) * epsilon
        x0[np.arange(L), seq] = one_minus_Z
        assert (PADDING_SYMBOL_INDEX == 0)
        x0[:, 0] = 0.
    return x0
