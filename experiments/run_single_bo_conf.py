import argparse
import random
import tensorflow as tf
import numpy as np
import poli
from gpflow import default_float
from poli import objective_factory
from poli.core.abstract_black_box import AbstractBlackBox
from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util.external_observer import ExternalObserver
from trieste.acquisition import ExpectedImprovement, ExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization, SearchSpaceType
from trieste.data import Dataset
from trieste.space import TaggedProductSearchSpace, Box, DiscreteSearchSpace
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.objectives.utils import mk_observer

from corel.optimization.lambo_optimizer import make_lambo_optimizer
from corel.optimization.latent_optimizer import ContinuousLatentSpaceParameterizationOptimizerFactory
from corel.optimization.pareto_frontier_explorer import make_pareto_frontier_explorer
from corel.optimization.randomized_pareto_frontier_explorer import make_randomized_pareto_frontier_explorer
from corel.optimization.simplex_optimizer import make_simplex_optimizer
from corel.protein_model import ProteinModel
from corel.trieste.custom_batch_acquisition_rule import CustomBatchEfficientGlobalOptimization
from corel.util.constants import PADDING_SYMBOL_INDEX, BATCH_SIZE
from corel.util.util import get_amino_acid_integer_mapping_from_info, transform_string_sequences_to_integer_arrays
from corel.weightings.hmm.hmm_factory import HMMFactory
from corel.weightings.vae.cbas.cbas_factory import CBASVAEFactory, CBASVAEWeightingFactory
from corel.weightings.vae.cbas.cbas_vae_wrapper import CBASVAEWrapper

# from experiments.config.problem_mappings import hmm_problem_model_mapping
# TODO: import this from experiments.config
hmm_problem_model_mapping = {
    "foldx_rfp_lambo": "/Users/rcml/corel/experiments/assets/hmms/rfp.hmm"
}

DEBUG = False
LOG_POST_PERFORMANCE_METRICS = False
TEMPLATE = f"python {__file__} "


def run_single_bo_conf(problem: str, max_blackbox_evaluations: int,
                       weighting_factory, optimizer_factory, seed: int = 0, batch_evaluations: int = 1):
    # make problem reproducible
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # build problem
    caller_info = dict()
    caller_info["DEBUG"] = DEBUG  # report if DEBUG flag is set
    caller_info[BATCH_SIZE] = batch_evaluations
    setup_info, blackbox_, train_x_, train_obj, run_info = objective_factory.create(problem, seed=seed,
                                                                                    caller_info=caller_info,
                                                                                    force_isolation=True,
                                                                                    observer=None)# ExternalObserver())
    # make tensorflow wrapper for problem
    L = setup_info.get_max_sequence_length()
    AA = len(setup_info.get_alphabet())
    if not setup_info.sequences_are_aligned():
        AA = AA + 1  # add one index for a padding symbol
    amino_acid_integer_mapping = get_amino_acid_integer_mapping_from_info(setup_info)
    integer_amino_acid_mapping = {amino_acid_integer_mapping[a]: a for a in amino_acid_integer_mapping.keys()}
    if not setup_info.sequences_are_aligned():
        assert(PADDING_SYMBOL_INDEX not in amino_acid_integer_mapping.values())

    def blackbox(x, context=None):
        x_ = x.numpy()
        seqs = np.array(["".join([integer_amino_acid_mapping[x_[n, i]] for i in range(x.shape[1]) if x[n,i] != PADDING_SYMBOL_INDEX]) for n in range(x.shape[0])])
        #raise NotImplementedError("How to make seqs a numpy array of lists of different length?")
        return tf.constant(blackbox_(seqs, context))

    train_x = transform_string_sequences_to_integer_arrays(train_x_, L, amino_acid_integer_mapping)

    # initialize logger (to track algorithm specific metrics)
    #initialize_logger(setup_info, method_factory.get_params(), seed=seed, run_id=run_info)

    # build Trieste problem
    observer = mk_observer(blackbox)
    amino_acid_space = DiscreteSearchSpace(tf.expand_dims(tf.range(AA), axis=-1))
    search_space = TaggedProductSearchSpace(L * [amino_acid_space])
    initial_data = Dataset(query_points=tf.squeeze(tf.constant(train_x)), observations=tf.constant(train_obj, dtype=default_float()))
    bo = BayesianOptimizer(observer, search_space)
    if train_obj.shape[1] == 1:
        # use expected improvement for single task objectives...
        ei_search_space = TaggedProductSearchSpace(L * [Box(lower=AA * [0.], upper=AA * [1.])])
        ei = ExpectedImprovement(search_space=ei_search_space)
    elif train_obj.shape[1] > 1:
        # ...and hypervolume EI for multi-task problems
        ei = ExpectedHypervolumeImprovement()
    else:
        raise RuntimeError("What kind of objective is that?!")
    rule = CustomBatchEfficientGlobalOptimization(optimizer=optimizer_factory(setup_info, batch_evaluations=batch_evaluations), builder=ei, num_query_points=batch_evaluations)
    weighting = weighting_factory.create(setup_info)
    #model = TrainableModelStack(*[(ProteinModel(weighting, AA=AA), 1) for _ in range(train_obj.shape[1])])
    model = ProteinModel(weighting, AA)
    result = bo.optimize(max_blackbox_evaluations, initial_data, model, acquisition_rule=rule)
    #blackbox_.terminate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-m", "--max_evaluations", type=int, default=32)
    parser.add_argument("-p", "--problem", type=str, default=poli.core.registry.get_problems()[0],
                        choices=poli.core.registry.get_problems())
    parser.add_argument("-b", "--batch_evaluations", type=int, default=16)
    args = parser.parse_args()
    tf.config.run_functions_eagerly(run_eagerly=True)
    #_call_run(**vars(args))
    #problem = "foldx_rfp_lambo"
    #optimizer_factory = make_lambo_optimizer
    #weighting_factory = HMMFactory(hmm_problem_model_mapping[problem], problem)
    problem = "gfp_cbas"
    optimizer_factory = lambda problem_info, batch_evaluations: ContinuousLatentSpaceParameterizationOptimizerFactory(problem_info, batch_size=batch_evaluations).create()
    weighting_factory = CBASVAEWeightingFactory()
    run_single_bo_conf(problem, 32, weighting_factory, optimizer_factory,
                       seed=0, batch_evaluations=16)
