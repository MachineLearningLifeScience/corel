__author__ = 'Simon Bartels'

import numpy as np
import tensorflow as tf

from corel.util.constants import PADDING_SYMBOL_INDEX
from corel.optimization.pareto_frontier_explorer import make_pareto_frontier_explorer, _seq_to_atom

"""
This optimization algorithm imitates LamBO. We just explore all single site mutations from the current Pareto front.
"""


def _get_best_of_single_site_mutations(seq, acquisition_function, trials):
    AA = acquisition_function._model.AA  # get from ac
    atom = _seq_to_atom(seq, AA)
    val = acquisition_function(atom)

    seq_ = seq.numpy().copy()
    #seq_ = seq_[seq_ != PADDING_SYMBOL_INDEX]
    padded_until = np.argmax(seq_ == PADDING_SYMBOL_INDEX)
    if padded_until == 0:
        padded_until = atom.shape[1]
    assert(tf.reduce_all(seq[0, padded_until:] == PADDING_SYMBOL_INDEX))
    for _ in range(trials):
        l = np.random.randint(0, padded_until)
        assert (PADDING_SYMBOL_INDEX == 0)
        for a in range(1, AA):
            seq_[0, l] = a
            val_ = acquisition_function(_seq_to_atom(seq_, AA))
            if val_ > val:
                seq = tf.constant(seq_.copy())
                val = val_
        seq_[0, l] = seq[0, l].numpy()
    return seq, val


def make_randomized_pareto_frontier_explorer(dataset=None, trials=10):
    get_best_of_single_site_mutations = lambda seq, acquisition_function: _get_best_of_single_site_mutations(seq, acquisition_function, trials)
    return make_pareto_frontier_explorer(dataset, get_best_of_single_site_mutations)
