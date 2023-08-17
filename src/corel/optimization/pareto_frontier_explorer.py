__author__ = 'Simon Bartels'

import numpy as np
import tensorflow as tf
from gpflow import default_float
from trieste.acquisition.multi_objective import non_dominated
from trieste.space import SearchSpaceType

from corel.util.constants import PADDING_SYMBOL_INDEX

"""
This optimization algorithm imitates LamBO. We just explore all single site mutations from the current Pareto front.
"""


def _get_best_of_single_site_mutations(seq, acquisition_function):
    AA = acquisition_function._model._models[0].AA  # get from ac
    atom = _seq_to_atom(seq, AA)
    val = acquisition_function(atom)

    seq_ = seq.numpy().copy()
    #seq_ = seq_[seq_ != PADDING_SYMBOL_INDEX]
    padded_until = np.argmax(seq_ == PADDING_SYMBOL_INDEX)
    if padded_until == 0:
        padded_until = atom.shape[1]
    assert(tf.reduce_all(seq[0, padded_until:] == PADDING_SYMBOL_INDEX))
    for l in range(padded_until):
        assert(PADDING_SYMBOL_INDEX == 0)
        for a in range(1, AA):
            seq_[0, l] = a
            val_ = acquisition_function(_seq_to_atom(seq_, AA))
            if val_ > val:
                seq = tf.constant(seq_.copy())
                val = val_
        seq_[0, l] = seq[0, l].numpy()
    return seq, val


def make_pareto_frontier_explorer(dataset=None, get_best_of_single_site_mutations=_get_best_of_single_site_mutations):
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
        # TODO: adapt for batch setting
        # TODO: test whether it's necessary to change the sign of the observations
        # seems not
        _, idx = non_dominated(observations_handle(acquisition_function))
        proposal_seqs = inputs_handle(acquisition_function)[idx]
        selected_seq, best_val = get_best_of_single_site_mutations(proposal_seqs[:1, ...], acquisition_function)
        for i in range(1, len(proposal_seqs)):
            s, v = get_best_of_single_site_mutations(proposal_seqs[i:i+1, ...], acquisition_function)
            if v > best_val:
                selected_seq = s
                best_val = v
        return selected_seq
    return pareto_frontier_explorer


def _seq_to_atom(x, AA):
    return tf.reshape(tf.one_hot(x, depth=AA, axis=-1, dtype=default_float()), [x.shape[0], 1, x.shape[1]*AA])
