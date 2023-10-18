import argparse
import random
import tensorflow as tf
import numpy as np
import poli
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

from corel.aligning_protein_model import AligningProteinModel
from corel.optimization.hmm_simplex_optimizer import make_hmm_simplex_optimizer
from corel.optimization.lambo_optimizer import make_lambo_optimizer
from corel.optimization.pareto_frontier_explorer import make_pareto_frontier_explorer
from corel.optimization.randomized_pareto_frontier_explorer import make_randomized_pareto_frontier_explorer
from corel.optimization.simplex_optimizer import make_simplex_optimizer
from corel.optimization.unaligned_batch_simplex_optimizer import make_unaligned_batch_simplex_optimizer
from corel.protein_model import ProteinModel
from corel.trieste.custom_batch_acquisition_rule import CustomBatchEfficientGlobalOptimization
from corel.util.constants import PADDING_SYMBOL_INDEX, BATCH_SIZE
from corel.util.util import get_amino_acid_integer_mapping_from_info, transform_string_sequences_to_integer_arrays
from corel.weightings.hmm.hmm_factory import HMMFactory
from corel.weightings.hmm.load_phmm import load_hmm
from corel.weightings.hmm.unaligned_hmm_weighting import UnalignedHMMWeighting
from experiments.config.problem_mappings import hmm_problem_model_mapping
from hmm_profile import reader

# TODO: import this from experiments.config
# hmm_problem_model_mapping = {
#     "foldx_rfp_lambo": "/Users/rcml/corel/experiments/assets/hmms/rfp.hmm"
# }

DEBUG = True
LOG_POST_PERFORMANCE_METRICS = False
TEMPLATE = f"python {__file__} "


def run_unaligned_hmm_optimizer_on_debug(seed=0):
    # make problem reproducible
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # build problem
    caller_info = dict()
    caller_info["DEBUG"] = DEBUG  # report if DEBUG flag is set
    AMINO_ACIDS = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "E",
        "Q",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]
    setup_info = ProblemSetupInformation("debug", 244, False, AMINO_ACIDS)
    class DebugBlackBox(AbstractBlackBox):
        def _black_box(self, x, context=None):
            return np.random.randn(x.shape[0], 2)
    blackbox_ = DebugBlackBox(setup_info)
    train_x_ = ["ARN", "DCEE"]
    train_obj = np.random.randn(2, 2)

    #train_x_ = train_x_[:4]
    #train_obj = train_obj[:4, ...]

    # make tensorflow wrapper for problem
    L = setup_info.get_max_sequence_length()
    AA = len(setup_info.get_alphabet())
    if not setup_info.sequences_are_aligned():
        AA = AA + 1  # add one index for a padding symbol
    amino_acid_integer_mapping = get_amino_acid_integer_mapping_from_info(setup_info)
    integer_amino_acid_mapping = {amino_acid_integer_mapping[a]: a for a in amino_acid_integer_mapping.keys()}
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
    initial_data = Dataset(query_points=tf.squeeze(tf.constant(train_x)), observations=tf.constant(train_obj))
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
    optimizer_factory = make_hmm_simplex_optimizer
    optimizer_function = optimizer_factory(setup_info, batch_evaluations=1)
    rule = CustomBatchEfficientGlobalOptimization(optimizer=optimizer_function, builder=ei, num_query_points=1)
    amino_acid_integer_mapping = get_amino_acid_integer_mapping_from_info(setup_info)
    with open("./assets/hmms/rfp.hmm") as f:
        hmm = reader.read_single(f)
        f.close()
    s0, T, em, extra_info_dict = load_hmm(hmm)
    # TODO: is the hmm alphabet the right one?
    weighting = UnalignedHMMWeighting(s0, T, em, hmm.metadata.alphabet, amino_acid_integer_mapping)
    #model = TrainableModelStack(*[(ProteinModel(weighting, AA=AA), 1) for _ in range(train_obj.shape[1])])
    model = AligningProteinModel(weighting, AA)
    max_blackbox_evaluations = 3
    result = bo.optimize(max_blackbox_evaluations, initial_data, model, acquisition_rule=rule)
    #blackbox_.terminate()


if __name__ == '__main__':
    tf.config.run_functions_eagerly(run_eagerly=True)
    run_unaligned_hmm_optimizer_on_debug()
