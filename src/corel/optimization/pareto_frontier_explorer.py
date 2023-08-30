__author__ = 'Simon Bartels'

import time

import numpy as np
import tensorflow as tf
from gpflow import default_float
from trieste.acquisition.multi_objective import non_dominated
from trieste.space import SearchSpaceType

from corel.util.constants import PADDING_SYMBOL_INDEX
from corel.util.k_best import KeepKBest

"""
This optimization algorithm imitates LamBO. We just explore all single site mutations from the current Pareto front.
"""


def _get_best_of_single_site_mutations(seq, acquisition_function, bestk: KeepKBest, position_function):
    AA = acquisition_function._model.AA  # get from ac

    seq_ = seq.numpy().copy()
    # no need to evaluate ac(seq) since we already know the function value anyway
    #val_ = acquisition_function(_seq_to_atom(seq_, AA))
    #bestk.new_val(val_, seq_)

    positions = position_function(seq)
    for l in positions:
        assert(PADDING_SYMBOL_INDEX == 0)
        for a in range(1, AA):
            seq_[0, l] = a
            val_ = acquisition_function(_seq_to_atom(seq_, AA))
            if bestk.new_val(val_, seq_) == 0:
                # if a mutation is the best, we take this as the starting point for further mutations
                seq = seq_.copy()
        # restore
        seq_[0, l] = seq[0, l]
    return bestk


def _padded_until(seq):
    #seq_ = seq_[seq_ != PADDING_SYMBOL_INDEX]
    padded_until = np.argmax(seq.numpy() == PADDING_SYMBOL_INDEX)
    if padded_until == 0:
        padded_until = seq.shape[1]
    assert(tf.reduce_all(seq[0, padded_until:] == PADDING_SYMBOL_INDEX))
    return padded_until


def _position_function(seq: tf.Tensor):
    return np.arange(_padded_until(seq))


def make_pareto_frontier_explorer(dataset=None, position_function=_position_function, batch_evaluations=1):
    if dataset is None:
        # observations_handle = lambda ac: tf.concat([ac._model._models[i].dataset.observations for i in range(len(ac._model._models))], axis=-1)
        # inputs_handle = lambda ac: ac._model._models[0].dataset.query_points
        observations_handle = lambda ac: ac._model.dataset.observations
        inputs_handle = lambda ac: ac._model.dataset.query_points
    else:
        # TODO: THIS DOES NOT WORK!
        raise NotImplementedError("this does not work! I need the most recent data!")
        observations_handle = lambda ac: dataset

    def pareto_frontier_explorer(search_space: SearchSpaceType, acquisition_function) -> tf.Tensor:
        _, idx = non_dominated(observations_handle(acquisition_function))
        proposal_seqs = inputs_handle(acquisition_function)[idx]
        bestk = KeepKBest(k=batch_evaluations, copy=lambda x: x.copy())
        print("mutating candidate 0 of " + str(len(proposal_seqs)))
        t = time.time()
        bestk = _get_best_of_single_site_mutations(proposal_seqs[:1, ...], acquisition_function, bestk, position_function)
        t = time.time() - t
        print("time: " + str(t))
        for i in range(1, len(proposal_seqs)):
            print("mutating candidate " + str(i) + " of " + str(len(proposal_seqs)))
            t = time.time()
            bestk = _get_best_of_single_site_mutations(proposal_seqs[i:i+1, ...], acquisition_function, bestk, position_function)
            t = time.time() - t
            print("time: " + str(t))
        #print("best acquisition value found: " + str(best_val))
        selected_seqs, vals = bestk.get()
        return tf.constant(np.concatenate(selected_seqs.tolist()))
    return pareto_frontier_explorer


def _seq_to_atom(x, AA):
    return tf.reshape(tf.one_hot(x, depth=AA, axis=-1, dtype=default_float()), [x.shape[0], 1, x.shape[1]*AA])
