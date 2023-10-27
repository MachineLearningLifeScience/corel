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

    def get_p0(padded_seq, hmm):
        if not problem_info.sequences_are_aligned():
            seq = padded_seq[padded_seq != PADDING_SYMBOL_INDEX]
        else:
            seq = padded_seq
        seq_to_int = np.array([hmm.index_map[seq[j]] for j in range(len(seq))])
        state_path = viterbi(seq_to_int, hmm.T, hmm.em, np.squeeze(hmm.s0))
        assert(PADDING_SYMBOL_INDEX == 0)
        frequencies = np.zeros(hmm.em.shape)
        for s in range(hmm.T.shape[0]):
            f = np.bincount(seq_to_int[np.where(state_path == s)])
            if len(f) == 0:
                f = hmm.em[s, :]
            frequencies[s, :len(f)] = f / np.sum(f)
        p0 = np.concatenate([np.zeros([hmm.T.shape[0], 1]), frequencies], axis=1)
        return p0



    def unaligned_batch_simplex_optimizer(search_space: SearchSpaceType, acquisition_function) -> tf.Tensor:
        assert(isinstance(acquisition_function._model, AligningProteinModel))
        hmm: UnalignedHMMWeighting = acquisition_function._model.distribution
        AA = search_space.upper[0].numpy() + 1
        if problem_info.sequences_are_aligned():
            assert(AA == len(problem_info.get_alphabet()))
        else:
            assert(AA - 1 == len(problem_info.get_alphabet()))

        assert(PADDING_SYMBOL_INDEX == 0)
        #p0 = np.concatenate([np.zeros([hmm.T.shape[0], 1]), hmm.em], axis=1)
        ##p0 = np.concatenate([np.zeros([hmm.T.shape[0], 1]), np.random.rand(*hmm.em.shape)], axis=1)
        # TODO: remove above line
        #p0 = p0 / np.sum(p0, axis=-1)[:, np.newaxis]  # TODO: should be normalized but isn't. Investigate!

        # TODO: a much better initial distribution is probably doing an encode/decode
        p0 = get_p0(padded_seq=inputs_handle(acquisition_function)[-1].numpy(), hmm=hmm)
        opt = _make_optimizer(p0.shape[0], AA, acquisition_function, p0)
        p = opt()
        # TODO: now we ideally get the viterbi path for all observed sequences and sample according to p
        padded_seq = inputs_handle(acquisition_function)[0].numpy()
        if not problem_info.sequences_are_aligned():
            seq = padded_seq[padded_seq != PADDING_SYMBOL_INDEX]
        else:
            seq = padded_seq
        seq_to_int = np.array([hmm.index_map[seq[j]] for j in range(len(seq))])
        state_path = viterbi(seq_to_int, hmm.T, hmm.em, hmm.s0)
        x = tf.expand_dims(tf.concat([tf.argmax(p[s]) for s in state_path], axis=0), axis=0)
        if not problem_info.sequences_are_aligned():
            x = tf.concat([x, PADDING_SYMBOL_INDEX * tf.ones([1, problem_info.get_max_sequence_length() - x.shape[1]], dtype=x.dtype)], axis=1)
        # TODO: keep in mind that this acquisition function optimizes over HMM emission probabilities!
        # TODO: this means that to evaluate the sequence we need to do something different!
        #print("best acquisition value found: " + str(best_val))
        return x  #tf.concat([x])
    return unaligned_batch_simplex_optimizer


