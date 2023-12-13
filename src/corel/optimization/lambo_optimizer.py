__author__ = 'Simon Bartels'

import time

import numpy as np
import tensorflow as tf
from gpflow import default_float
from trieste.acquisition.multi_objective import non_dominated
from trieste.space import SearchSpaceType

from corel.optimization.pareto_frontier_explorer import _padded_until, _seq_to_atom
from corel.util.constants import PADDING_SYMBOL_INDEX
from corel.util.k_best import KeepKBest

"""
This optimization algorithm imitates LamBO. We just explore all single site mutations from the current Pareto front.
"""


def mutate_candidate(seq, acquisition_function, copy=lambda x: x.copy(), trials=1):
    AA = acquisition_function._model.AA  # get from ac
    bestk = KeepKBest(1, copy=copy)
    positions = np.random.randint(0, _padded_until(seq), trials)

    # no need to evaluate ac(seq) since we already know the function value anyway
    seq_ = seq.numpy().copy()
    #val_ = acquisition_function(_seq_to_atom(seq_, AA))
    #bestk.new_val(val_, seq_)

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
    return bestk.get()[0][0]


def make_lambo_optimizer(dataset=None, batch_evaluations=1):
    if dataset is None:
        # observations_handle = lambda ac: tf.concat([ac._model._models[i].dataset.observations for i in range(len(ac._model._models))], axis=-1)
        # inputs_handle = lambda ac: ac._model._models[0].dataset.query_points
        observations_handle = lambda ac: ac._model.dataset.observations
        inputs_handle = lambda ac: ac._model.dataset.query_points
    else:
        # TODO: THIS DOES NOT WORK!
        raise NotImplementedError("this does not work! I need the most recent data!")
        observations_handle = lambda ac: dataset
    # TODO: if _model / acquisition function fails, remove proposals
    def lambo_optimizer(search_space: SearchSpaceType, acquisition_function) -> tf.Tensor:
        _, idx_ = non_dominated(observations_handle(acquisition_function))
        proposal_seqs = inputs_handle(acquisition_function)[idx_]
        selected_seqs = []
        for _round, i in enumerate(np.random.randint(0, proposal_seqs.shape[0], batch_evaluations)):
            print("mutating candidate " + str(_round+1) + " of " + str(batch_evaluations))
            t = time.time()
            best = mutate_candidate(proposal_seqs[i:i+1, ...], acquisition_function)
            selected_seqs.append(best)
            t = time.time() - t
            print("time: " + str(t))
        return tf.constant(np.concatenate(selected_seqs))
    return lambo_optimizer
