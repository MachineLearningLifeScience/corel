__author__ = 'Simon Bartels'
import tensorflow as tf
from trieste.acquisition.multi_objective import non_dominated
from trieste.space import SearchSpaceType

from corel.util.util import seq_to_atom

"""
This optimization algorithm imitates LamBO. We just explore all single site mutations from the current Pareto front.
"""


def make_pareto_frontier_explorer(dataset=None):
    if dataset is None:
        observations_handle = lambda ac: ac._model._models[0].dataset
    else:
        # TODO: THIS DOES NOT WORK!
        raise NotImplementedError("this does not work! I need the most recent data!")
        observations_handle = lambda ac: dataset

    def pareto_frontier_explorer(search_space: SearchSpaceType, acquisition_function) -> tf.Tensor:
        # TODO: adapt for batch setting
        # TODO: test whether it's necessary to change the sign of the observations
        # seems not
        _, idx = non_dominated(observations_handle(acquisition_function).observations)
        proposal_seqs = observations_handle(acquisition_function).query_points[idx]
        selected_seq, best_val = _get_best_of_single_site_mutations(proposal_seqs[0], acquisition_function)
        for i in range(1, len(proposal_seqs)):
            s, v = _get_best_of_single_site_mutations(proposal_seqs[i], acquisition_function)
            if v > best_val:
                selected_seq = s
                best_val = v
        return selected_seq
    return pareto_frontier_explorer


def _get_best_of_single_site_mutations(seq, acquisition_function):
    raise NotImplementedError("TODO: check whether problem is aligned and if not, take care of padding symbol!")
    atom = seq_to_atom(seq)
    val = acquisition_function(atom)

    seq_ = seq.numpy().copy()
    for l in range(P.shape[1]):
        for a in range(1, P.shape[2]):
            seq_[0, l] = a
            val_ = acquisition_function(seq_to_atom(seq_))
            if val_ > val:
                seq = tf.constant(seq_.copy())
                val = val_
        seq_[0, l] = seq[0, l].numpy()
    return seq, val
