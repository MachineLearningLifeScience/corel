import random
import tensorflow as tf
import numpy as np
from poli import objective_factory
from poli.core.problem_setup_information import ProblemSetupInformation
from trieste.acquisition import ExpectedImprovement, ExpectedHypervolumeImprovement
from trieste.acquisition.rule import EfficientGlobalOptimization, SearchSpaceType
from trieste.data import Dataset
from trieste.space import TaggedProductSearchSpace, Box, DiscreteSearchSpace
from trieste.bayesian_optimizer import BayesianOptimizer
from trieste.objectives.utils import mk_observer

from corel.optimization.pareto_frontier_explorer import make_pareto_frontier_explorer
from corel.optimization.randomized_pareto_frontier_explorer import make_randomized_pareto_frontier_explorer
from corel.protein_model import ProteinModel
from corel.trieste.custom_batch_acquisition_rule import CustomBatchEfficientGlobalOptimization
from corel.util.constants import PADDING_SYMBOL_INDEX, BATCH_SIZE
from corel.util.util import get_amino_acid_integer_mapping_from_info
from corel.weightings.hmm.hmm_factory import HMMFactory

DEBUG = False
LOG_POST_PERFORMANCE_METRICS = False
TEMPLATE = f"python {__file__} "


def run_single_bo_conf(problem: str, max_blackbox_evaluations: int,
                       weighting_factory, optimizer_factory, seed: int = 0, batch_evaluations: int = 1):
    # make problem reproducible
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

    caller_info = dict()
    caller_info["DEBUG"] = DEBUG  # report if DEBUG flag is set
    caller_info[BATCH_SIZE] = batch_evaluations
    if not DEBUG:
        setup_info, blackbox_, train_x_, train_obj, run_info = objective_factory.create(problem, seed=seed,
                                                                                        caller_info=caller_info)
    else:
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
        setup_info = ProblemSetupInformation("FOLDX_RFP", 244, False, AMINO_ACIDS)
        blackbox_ = lambda s, context=None: np.random.randn(s.shape[0], 2)
        train_x_ = ["ARN", "DCEE"]
        train_obj = np.random.randn(2, 2)

    #train_x_ = train_x_[:4]
    #train_obj = train_obj[:4, ...]

    L = setup_info.get_max_sequence_length()
    AA = len(setup_info.get_alphabet())
    if not setup_info.sequences_are_aligned():
        AA = AA + 1  # add one index for a padding symbol
    amino_acid_integer_mapping = get_amino_acid_integer_mapping_from_info(setup_info)
    integer_amino_acid_mapping = {amino_acid_integer_mapping[a]: a for a in amino_acid_integer_mapping.keys()}
    # TODO: move this code to utility function
    assert(PADDING_SYMBOL_INDEX not in amino_acid_integer_mapping.values())
    train_x = np.zeros([len(train_x_), L], dtype=int)
    for i in range(len(train_x_)):
        seq = train_x_[i]
        if seq == "GEELIKENMHMKLYMEGTVNNHHFKCTTEGEGKPYEGTQTQRIKVVEGGPLPFAFDILATCFSKTFINHTQGIPDFFKQSFPEGFTWERVTTYEDGGVLTVTQDTSLQDGCLIYNVKLRGVNFPSNGPVMQKKTLGWEATTETLYPADGGLEGRCDMALXLVGGGHLHCNLKTTYRSXKPAKNLKMPGVYFVDRRLERIKEADNETYVEQHEVAVARYCDLPSKL":
            # above seq is problematic due to an X in position 159
            # according to PDB this sequence has the most (98%) similarity: SKGEELIKENMHMKLYMEGTVNNHHFKCTTEGEGKPYEGTQTQRIKVVEGGPLPFAFDILATCFMYGSKTFINHTQGIPDFFKQSFPEGFTWERVTTYEDGGVLTVTQDTSLQDGCLIYNVKLRGVNFPSNGPVMQKKTLGWEATTETLYPADGGLEGRCDMALKLVGGGHLHCNLKTTYRSKKPAKNLKMPGVYFVDRRLERIKEADNETYVEQHEVAVARYCDLPSKL
            # I've removed the initial SK and some MYG in the middle
            # Then it's really just a K in position 159 which is different.
            #seq[159] = 'K'  # strings are immutable
            seq = "GEELIKENMHMKLYMEGTVNNHHFKCTTEGEGKPYEGTQTQRIKVVEGGPLPFAFDILATCFSKTFINHTQGIPDFFKQSFPEGFTWERVTTYEDGGVLTVTQDTSLQDGCLIYNVKLRGVNFPSNGPVMQKKTLGWEATTETLYPADGGLEGRCDMALKLVGGGHLHCNLKTTYRSKKPAKNLKMPGVYFVDRRLERIKEADNETYVEQHEVAVARYCDLPSKL"
        len_seq = len(seq)
        train_x[i, :len_seq] = np.array([amino_acid_integer_mapping[a] for a in seq])
        train_x[i, len_seq:] = PADDING_SYMBOL_INDEX
    train_x = tf.constant(train_x)

    def blackbox(x, context=None):
        x_ = x.numpy()
        seqs = np.array(["".join([integer_amino_acid_mapping[x_[n, i]] for i in range(x.shape[1]) if x[n,i] != PADDING_SYMBOL_INDEX]) for n in range(x.shape[0])])
        #raise NotImplementedError("How to make seqs a numpy array of lists of different length?")
        return tf.constant(blackbox_(seqs, context))

    #initialize_logger(setup_info, method_factory.get_params(), seed=seed, run_id=run_info)

    observer = mk_observer(blackbox)
    amino_acid_space = DiscreteSearchSpace(tf.expand_dims(tf.range(AA), axis=-1))
    search_space = TaggedProductSearchSpace(L * [amino_acid_space])
    initial_data = Dataset(query_points=tf.squeeze(tf.constant(train_x)), observations=tf.constant(train_obj))
    bo = BayesianOptimizer(observer, search_space)
    #ei_search_space = TaggedProductSearchSpace(L * [Box(lower=AA * [0.], upper=AA * [1.])])
    #ei = ExpectedImprovement(search_space=ei_search_space)
    ei = ExpectedHypervolumeImprovement()
    rule = CustomBatchEfficientGlobalOptimization(optimizer=optimizer_factory(batch_evaluations=batch_evaluations), builder=ei, num_query_points=batch_evaluations)
    weighting = weighting_factory.create(setup_info)
    #model = TrainableModelStack(*[(ProteinModel(weighting, AA=AA), 1) for _ in range(train_obj.shape[1])])
    model = ProteinModel(weighting, AA)
    result = bo.optimize(max_blackbox_evaluations, initial_data, model, acquisition_rule=rule)
    #blackbox_.terminate()


if __name__ == '__main__':
    tf.config.run_functions_eagerly(run_eagerly=True)
    problem = "FOLDX_RFP"
    problem_model_mapping = {
        "FOLDX_RFP": "./assets/hmms/rfp.hmm"
    }
    optimizer_factory = make_randomized_pareto_frontier_explorer
    run_single_bo_conf(problem, 32, HMMFactory(problem_model_mapping[problem]), optimizer_factory,
                       seed=0, batch_evaluations=16)
    exit()
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-m", "--max_evaluations", type=int, default=5)
    parser.add_argument("-p", "--problem", type=str, default=poli.core.registry.get_problems()[0],
                        choices=poli.core.registry.get_problems())
    args = parser.parse_args()
    #_call_run(**vars(args))
