import argparse
import random
from typing import Callable
import numpy as np
import tensorflow as tf

from trieste.data import Dataset
from trieste.space import TaggedProductSearchSpace
from trieste.space import Box
from trieste.space import DiscreteSearchSpace
from trieste.acquisition import ExpectedImprovement
from trieste.acquisition import ExpectedHypervolumeImprovement
from trieste.objectives.utils import mk_observer
from trieste.objectives.utils import mk_multi_observer
from trieste.bayesian_optimizer import BayesianOptimizer

from poli import objective_factory
from poli.core.registry import set_observer
from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util.external_observer import ExternalObserver
from corel.optimization.lambo_optimizer import make_lambo_optimizer
from corel.protein_model import ProteinModel

from corel.util.constants import ALGORITHM, BATCH_SIZE, PADDING_SYMBOL_INDEX, SEED, STARTING_N, MODEL
from corel.trieste.custom_batch_acquisition_rule import CustomBatchEfficientGlobalOptimization
from corel.util.util import get_amino_acid_integer_mapping_from_info
from corel.util.util import transform_string_sequences_to_integer_arrays
from corel.protein_model import ProteinModel
from corel.weightings.hmm.hmm_factory import HMMFactory
from corel.weightings.vae.base.vae_factory import VAEFactory

problem_model_mapping = { # TODO: these filenames should come from a config and are problem specific
    "foldx_rfp_lambo": {
        "hmm": "./experiments/assets/hmms/rfp.hmm",
        "vae": "./results/models/vae_z_2_rfp_fam.ckpt"
        },
    "rank_gfp": {}, # TODO: implement
}

AVAILABLE_WEIGHTINGS = [HMMFactory, VAEFactory]
# number of available observation from cold (0.) to warm (250+) start
AVAILABLE_SEQUENCES_N = [1, 16, 512]

RFP_PROBLEM_NAMES = list(problem_model_mapping.keys())

LOG_POST_PERFORMANCE_METRICS = False


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return


def get_acquisition_function_from_y(y: tf.Tensor, L: int, AA: int) -> object:
    """
    Given the observations (single or MT) get acquisition function.
    Returns:
        acquisition function object
    """
    if y.shape[1] == 1:
        search_space = TaggedProductSearchSpace(
            L * [Box(lower=AA*[0.], upper=AA*[1.])]
            )
        ei = ExpectedImprovement(search_space=search_space)
    elif y.shape[1] > 1:
        ei = ExpectedHypervolumeImprovement()
    else:
        raise ValueError(f"Objective misspecified! \nIncoherent y.shape={y.shape}")
    return ei


def cold_start_experiment(seed: int, budget: int, batch: int, n_allowed_observations: int, problem: str, p_factory: object):
    if not problem:
        raise ValueError("Specify Problem!")
    set_seeds(seed)
    caller_info = {
        BATCH_SIZE: batch,
        SEED: seed,
        STARTING_N: n_allowed_observations,
        MODEL: p_factory.__class__.__name__,
        ALGORITHM: "COREL",
    }
    problem_info, _f, _x0, _y0, run_info = objective_factory.create(
        name=problem,
        seed=seed,
        caller_info=caller_info,
        observer=ExternalObserver(),
        force_register=True,
        parallelize=False, # TODO: enable parallelize
        # num_workers=4,
        # batch_size=1,
    )
    # subselect initial data and observations
    _x0 = _x0[:n_allowed_observations] 
    y0 = _y0[:n_allowed_observations]

    L = problem_info.get_max_sequence_length()
    AA = len(problem_info.get_alphabet())
    if not problem_info.sequences_are_aligned():
        AA += 1 # account for padding token

    aa_int_mapping = get_amino_acid_integer_mapping_from_info(problem_info)
    int_aa_mapping = {aa_int_mapping[a]: a for a in aa_int_mapping.keys()}
    assert PADDING_SYMBOL_INDEX not in aa_int_mapping.values() # TODO: why?

    def f_wrapper(x, f=_f, aa_mapping: dict=int_aa_mapping, context=None):
        _x = x.numpy()
        seqs = np.array([
            "".join([aa_mapping[_x[n, i]] for i in range(x.shape[1]) if x[n,i] != PADDING_SYMBOL_INDEX]) 
                for n in range(x.shape[0])]
        )
        return tf.constant(f(seqs, context))

    X_train = transform_string_sequences_to_integer_arrays(_x0, L, aa_int_mapping)
    aa_space = DiscreteSearchSpace(tf.expand_dims(tf.range(AA), axis=-1))
    search_space = TaggedProductSearchSpace(L*[aa_space])
    
    tr_observer = mk_observer(f_wrapper)  # add encoding and context to black-box observation
    dataset_t0 = Dataset(query_points=tf.constant(X_train), observations=tf.constant(y0))
    ei = get_acquisition_function_from_y(y0, L=L, AA=AA)
    optimizer_factory = make_lambo_optimizer # TODO: account for other optimizers
    rule = CustomBatchEfficientGlobalOptimization(
            optimizer=optimizer_factory(batch_evaluations=batch),
            builder=ei,
            num_query_points=batch)
    weighting = p_factory.create(problem_info)
    model = ProteinModel(weighting, AA)
    bo = BayesianOptimizer(tr_observer, search_space)
    result = bo.optimize(num_steps=budget, datasets=dataset_t0, models=model, acquisition_rule=rule)
    return result


if __name__ == "__main__":
    # NOTE: this is currently the RFP experiment, other experiments require other experiment run scripts
    parser = argparse.ArgumentParser(description="Experiment Specifications Cold to Warm-Start")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed for experiments.")
    parser.add_argument("-m", "--max_evaluations", type=int, default=100, help="Optimization budget, number of possible observations.")
    parser.add_argument("-p", "--problem", type=str, choices=list(problem_model_mapping.keys()), default=list(problem_model_mapping.keys())[0], help="Problem description as string key.")
    parser.add_argument("-b", "--batch", type=int, default=1)
    parser.add_argument("-n", "--number_observations", type=int, choices=AVAILABLE_SEQUENCES_N, default=AVAILABLE_SEQUENCES_N[-1])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-w", "--weighting", type=str, choices=AVAILABLE_WEIGHTINGS, default=AVAILABLE_WEIGHTINGS[0])
    args = parser.parse_args()
    
    tf.config.run_functions_eagerly(run_eagerly=True)
    model_key = args.weighting.__name__.lower()[:3]
    result = cold_start_experiment(
        problem=args.problem, 
        seed=args.seed, 
        budget=args.max_evaluations, 
        batch=args.batch, 
        n_allowed_observations=args.number_observations,
        p_factory=args.weighting(problem_model_mapping[args.problem][model_key], args.problem)
    )
    if args.verbose:
        print(f"Optimal result: {result}")