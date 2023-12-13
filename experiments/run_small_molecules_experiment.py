"""Bayesian optimization of small molecule properties using continuous relaxations

This script runs the Bayesian optimization experiment for SELFIES molecules
on Zinc250k, using a VAE for weighting the Hellinger kernel.
"""

import numpy as np
import tensorflow as tf

from trieste.acquisition import ExpectedHypervolumeImprovement
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.bayesian_optimizer import OptimizationResult
from trieste.data import Dataset
from trieste.objectives.utils import mk_observer
from trieste.space import TaggedProductSearchSpace
from trieste.space import DiscreteSearchSpace

from poli import objective_factory
from poli.core.multi_objective_black_box import MultiObjectiveBlackBox

from corel.trieste.custom_batch_acquisition_rule import (
    CustomBatchEfficientGlobalOptimization,
)
from corel.optimization.lambo_optimizer import make_lambo_optimizer

from corel.weightings.vae.small_molecules.vae_selfies import load_vae, VAESelfies
from corel.small_molecule_model import SmallMoleculeModel
from corel.util.small_molecules.data import load_zinc_250k_alphabet


def main() -> OptimizationResult:
    """Runs the optimization experiment for SELFIES molecules on Zinc250k"""

    # Defining some hyperparameters:
    budget = ...
    max_sequence_length = ...
    batch_size = ...

    # Defining the search space:
    alphabet = load_zinc_250k_alphabet()
    alphabet_length = n_categories = len(alphabet)
    aa_space = DiscreteSearchSpace(tf.expand_dims(tf.range(alphabet_length), axis=-1))
    search_space = TaggedProductSearchSpace(max_sequence_length * [aa_space])

    # Defining the function to optimize (i.e. the trieste observer)
    problem_info, f_qed, *_ = objective_factory.create(name="rdkit_qed")
    _, f_logp, *_ = objective_factory.create(name="rdkit_logp")

    f_ = MultiObjectiveBlackBox(
        problem_info, batch_size=None, objective_functions=[f_qed, f_logp]
    )

    def f_wrapper(x, f=f_, aa_mapping: dict = alphabet, context=None):
        _x = x.numpy()
        seqs = np.array(
            [
                "".join(
                    [
                        aa_mapping[_x[n, i]]
                        for i in range(x.shape[1])
                        if x[n, i] != PADDING_SYMBOL_INDEX
                    ]
                )
                for n in range(x.shape[0])
            ]
        )
        seqs = np.atleast_1d(seqs)
        # TODO: add model parameters as context to be tracked by observer here?
        return tf.constant(f(seqs, context))

    trieste_observer = mk_observer(f)

    # Defining the initial dataset
    # The initial x values (expected to be integer tensors [b, L])
    x_0 = None

    # Their corresponding y values (expected to be float tensors [b, 2])
    y_0 = None
    initial_dataset = Dataset(query_points=x_0, observations=y_0)

    # Defining the acquisition rule

    # A trieste optimizer is in charge of maximizing the
    # acquisition function. In our case, we use the same
    # algorithm as LAMBO, which essentially explores all
    # single site mutations from the current Pareto front.
    # TODO: we'll have to re-implement this optimizer for
    # small molecules instead.
    trieste_optimizer = make_lambo_optimizer(batch_evaluations=batch_size)

    # The acquisition function is just Expected Hypervolume Improvement
    acquisition_function = ExpectedHypervolumeImprovement()

    # The acquisition rule puts them together
    acquisition_rule = CustomBatchEfficientGlobalOptimization(
        builder=acquisition_function,
        optimizer=trieste_optimizer,
        num_query_points=batch_size,
    )

    # Defining the model

    # The weighting can be a VAE or an HMM. In the case of small molecules,
    # we use the VAESeflies implemented at:
    # corel.weightings.vae.small_molecules.vae_selfies
    # We expect the weighting to be a callable from tensors to
    # probability tensors over the sequence length (and alphabet).
    # TODO: is this really what we expect? Ask Simon.
    # (Check AbstractWeighting for more details).
    weighting: VAESelfies = load_vae()
    model = SmallMoleculeModel(weighting, n_categories)

    bo = BayesianOptimizer(trieste_observer, search_space)
    result = bo.optimize(
        num_steps=budget,
        datasets=initial_dataset,
        models=model,
        acquisition_rule=acquisition_rule,
    )

    return result
