import argparse
import logging
import random
from inspect import signature
from logging import info
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from poli import objective_factory
from poli.core.registry import set_observer
from trieste.acquisition import (ExpectedHypervolumeImprovement,
                                 ExpectedImprovement)
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.data import Dataset
from trieste.objectives.utils import mk_observer
from trieste.space import Box, DiscreteSearchSpace, TaggedProductSearchSpace

from corel.lvm_model import LVMModel
from corel.observers.poli_base_logger import PoliBaseMlFlowObserver
from corel.optimization.latent_optimizer import \
    ContinuousLatentSpaceParameterizationOptimizerFactory
from corel.protein_model import ProteinModel
from corel.trieste.custom_batch_acquisition_rule import \
    CustomBatchEfficientGlobalOptimization
from corel.util.constants import ALGORITHM, BATCH_SIZE, MODEL, SEED, STARTING_N
from corel.util.util import (get_amino_acid_integer_mapping_from_info,
                             transform_string_sequences_to_integer_arrays)
from corel.weightings.vae.cbas import CBASVAEWeightingFactory

tf.config.run_functions_eagerly(True)

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s", 
    level=logging.INFO, 
    datefmt="%Y-%m-%d %H:%M:%S"
    )

TRACKING_URI = Path(__file__).parent.parent / "results" / "mlruns"


AVAILABLE_WEIGHTINGS = [CBASVAEWeightingFactory]
# number of available observation from cold (0.) to warm (250+) start
AVAILABLE_SEQUENCES_N = [3, 16, 50, 512]

PROBLEM_NAMES = ["gfp_cbas_gp", "gfp_cbas_elbo"]

MODEL_CLASS = {
        ProteinModel.__name__: ProteinModel, 
        LVMModel.__name__: LVMModel
        }

LOG_POST_PERFORMANCE_METRICS = False


def set_seeds(seed: int) -> None:
    info(f"Setting seed={seed}")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return


def standardize(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = np.mean(y)
    std = np.std(y)
    std_y = (y - mean) / std
    return std_y, mean, std


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


def cold_start_gfp_experiment(seed: int, budget: int, batch: int, n_allowed_observations: int, problem: str, p_factory: object, model_class: object):
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
    observer = None
    observer = PoliBaseMlFlowObserver(TRACKING_URI)
    info("Invoking objective factory create")
    problem_info, _f, _x0, _y0, run_info = objective_factory.create(
        name=problem,
        seed=seed,
        caller_info=caller_info,
        batch_size=batch, # determines return shape of y0
        observer=observer,
        force_register=True,
        parallelize=False,
        problem_type=problem.split("_")[-1]
    )
    # subselect initial data and observations
    x0 = _x0[:batch] # corresponds to sequences for which f(x0) was computed
    y0 = _y0[:batch]

    # apply standardization
    y0, y_mu, y_sigma = standardize(y0)

    L = problem_info.get_max_sequence_length()
    AA = len(problem_info.get_alphabet())

    aa_int_mapping = get_amino_acid_integer_mapping_from_info(problem_info)
    int_aa_mapping = {aa_int_mapping[a]: a for a in aa_int_mapping.keys()}

    info("Load weighting distributions")
    weighting = p_factory().create(problem_info=problem_info)
    info(f"Set up model: {model_class.__name__}")
    if "L" in signature(model_class.__init__).parameters.keys():
        info("Querying available unlabelled data")
        init_data = weighting.get_training_data()
        # TODO: for DEV use only first 100 sequences
        init_data = init_data[:100] # FIXME
        init_data_int = transform_string_sequences_to_integer_arrays(init_data, L, aa_int_mapping)
        model = model_class(weighting, AA=AA, L=L, unlabelled_data=init_data_int)
    else:
        model = model_class(weighting, AA)

    def f_wrapper(x, f=_f, aa_mapping: dict=int_aa_mapping, model=model, context=None, problem=problem, y_mu=y_mu, y_sigma=y_sigma):
        _x = x.numpy()
        # convert int tensor to AA strings
        sequences = np.array([
            "".join([aa_mapping[_x[n, i]] for i in range(x.shape[1])]) 
                for n in range(x.shape[0])]
        )
        # log model parameters as metrics via a context
        if hasattr(model, 'get_context') and callable(model.get_context):
            context = model.get_context()
        seqs = np.array([list(_s) for _s in sequences])
        f_batch = f(seqs, context) # batched calls
        f_val = (f_batch - y_mu) / y_sigma # standardize observations
        return tf.constant(f_val)

    X_train = transform_string_sequences_to_integer_arrays(x0, L, aa_int_mapping)
    aa_space = DiscreteSearchSpace(tf.expand_dims(tf.range(AA), axis=-1))
    search_space = TaggedProductSearchSpace(L*[aa_space])
    
    tr_observer = mk_observer(f_wrapper)  # add encoding and context to black-box observation
    dataset_t0 = Dataset(query_points=tf.constant(X_train), observations=tf.constant(y0))
    ei = get_acquisition_function_from_y(y0, L=L, AA=AA)
    optimizer_factory = ContinuousLatentSpaceParameterizationOptimizerFactory(problem_info=problem_info, batch_size=batch)
    info("Setup batched EGO")
    rule = CustomBatchEfficientGlobalOptimization(
            optimizer=optimizer_factory.create(),
            builder=ei,
            num_query_points=batch)
    bo = BayesianOptimizer(tr_observer, search_space)
    info("Running trieste BO optimize routine")
    result = bo.optimize(num_steps=budget, datasets=dataset_t0, models=model, acquisition_rule=rule)
    if observer is not None:
        info("Terminating observer")
        observer.finish()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Specifications Cold to Warm-Start")
    parser.add_argument("-s", "--seed", type=int, default=0, help="Random seed for experiments.")
    parser.add_argument("-m", "--max_evaluations", type=int, default=100, help="Optimization budget, number of possible observations.")
    parser.add_argument("-p", "--problem", type=str, choices=PROBLEM_NAMES, default=PROBLEM_NAMES[0], help="Problem description as string key.")
    parser.add_argument("-b", "--batch", type=int, default=10)
    parser.add_argument("-n", "--number_observations", type=int, choices=AVAILABLE_SEQUENCES_N, default=AVAILABLE_SEQUENCES_N[-1])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-w", "--weighting", type=str, choices=AVAILABLE_WEIGHTINGS, default=AVAILABLE_WEIGHTINGS[0])
    parser.add_argument("--model", type=str, choices=MODEL_CLASS.keys(), default=MODEL_CLASS.get(LVMModel.__name__))
    args = parser.parse_args()
    info(f"Running GFP experiment: {args.problem}\n Model: {args.model} weighting: {args.weighting}\nbudget={args.max_evaluations} batch_size={args.batch} seed={args.seed}")
    result = cold_start_gfp_experiment(
        problem=args.problem, 
        seed=args.seed, 
        budget=args.max_evaluations, 
        batch=args.batch, 
        n_allowed_observations=args.number_observations,
        p_factory=args.weighting,
        model_class=args.model,
    )
    if args.verbose:
        info(f"Optimal result: {result}")