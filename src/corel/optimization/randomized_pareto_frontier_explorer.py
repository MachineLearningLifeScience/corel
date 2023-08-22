__author__ = 'Simon Bartels'

import numpy as np
import tensorflow as tf

from corel.optimization.pareto_frontier_explorer import make_pareto_frontier_explorer, _seq_to_atom, _padded_until

"""
This optimization algorithm imitates LamBO. We just explore all single site mutations from the current Pareto front.
"""


def make_randomized_pareto_frontier_explorer(dataset=None, batch_evaluations=1, trials=10):
    def _position_function(seq: tf.Tensor):
        return np.random.randint(0, _padded_until(seq), trials)

    return make_pareto_frontier_explorer(dataset, _position_function, batch_evaluations=batch_evaluations)
