"""This script runs the baselines on the different objectives.

Baselines to run:
- Random mutations of the pareto front
- NSGA-II (for the multi-objective problems)

Objective functions to optimize:
- "foldx_rfp_lambo" (x)
- "foldx_stability_and_sasa" (x)

To run this script, you'll need to:

1. Set up poli.
    - Clone poli and install it:
      pip install git+https://github.com/MachineLearningLifeScience/poli.git@dev
    - Make sure your poli__lambo environment works well,
      this will imply cloning and installing lambo locally.
    - For "foldx_stability_and_sasa", you should run this
      script from an environment that has pdb-tools and biopython
      installed.

2. Set up poli-baselines.
    - Clone poli-baselines and install it:
      pip install git+https://github.com/MachineLearningLifeScience/poli-baselines
"""
from typing import Tuple, Literal, Union
from pathlib import Path
import warnings
import argparse

warnings.filterwarnings("ignore", module="Bio")

import numpy as np

# from poli.objective_repository import RFPWrapperFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.objective_repository import RFPFoldXStabilityAndSASAProblemFactory
from poli.core.abstract_black_box import AbstractBlackBox

from poli_baselines.solvers import (
    DiscreteNSGAII,
    RandomMutation,
)

from corel.observers.poli_base_logger import PoliBaseMlFlowObserver
from corel.observers.poli_lambo_logger import PoliLamboLogger
from corel.util.constants import (
    ALGORITHM,
    BATCH_SIZE,
    SEED,
    STARTING_N,
    MODEL,
)
from corel.util.util import transform_string_sequences_to_string_arrays

from lambo import __file__ as lambo_project_root_file

LAMBO_PROJECT_ROOT = Path(lambo_project_root_file).parent.parent.resolve()
LAMBO_FOLDX_ASSETS = LAMBO_PROJECT_ROOT / "lambo" / "assets" / "foldx"
PROJECT_ROOT_DIR = Path(__file__).parent.parent.resolve()
TRACKING_URI = f"file:/{PROJECT_ROOT_DIR}/results/mlruns"

AVAILABLE_SEQUENCES_N = [6, 16, 50]
BASELINE_NAMES = ["RandomMutation", "DiscreteNSGAII"]


def instantiate_baseline(
    baseline_name: Literal["RandomMutation", "FixedLengthGeneticAlgorithm"],
    f: AbstractBlackBox,
    x0: np.ndarray,
    y0: np.ndarray,
    n_mutations: int,
    batch_size: int,
    population_size: int,
    prob_mutations: float = 0.5,
):
    if baseline_name == "RandomMutation":
        baseline = RandomMutation(
            black_box=-f,
            x0=x0,
            y0=-y0,
            n_mutations=n_mutations,
            top_k=batch_size,
            batch_size=16,
            greedy=True,
        )
    elif baseline_name == "DiscreteNSGAII":
        baseline = DiscreteNSGAII(
            black_box=-f,
            x0=x0,
            y0=-y0,
            population_size=population_size,
            initialize_with_x0=True,
            num_mutations=n_mutations,
        )
    else:
        raise ValueError(f"Unknown baseline {baseline_name}")

    return baseline


def prepare_data_for_experiment(
    x0: np.ndarray,
    y0: np.ndarray,
    n_allowed_observations: int,
    problem_info: ProblemSetupInformation,
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepares data for the "foldx_rfp_lambo" experiment."""
    # subselect initial data and observations
    x0 = x0[:n_allowed_observations]
    y0 = y0[:n_allowed_observations]

    L = problem_info.get_max_sequence_length()
    if L == np.inf:
        L = (
            max(len(_x) for _x in x0) + 25
        )  # make length value deterministic if ill-defined # NOTE: in base RFP L=425

    X_train = transform_string_sequences_to_string_arrays(x0, L)

    return X_train, y0


def instantiate_black_box(
    problem_name: Literal["rfp_foldx_stability_and_sasa", "foldx_rfp_lambo"],
    caller_info: dict,
    n_allowed_observations: int,
    batch_size: int,
    seed: int,
    num_workers: int = 4,
):
    assets_pdb_paths = list(LAMBO_FOLDX_ASSETS.glob("*/wt_input_Repair.pdb"))

    # This is mostly taken from run_cold_warm_start_experiments_rfp_bo.py
    if problem_name == "rfp_foldx_stability_and_sasa":
        if (
            n_allowed_observations == 1
        ):  # cold-start problem: optimize w.r.t. 1 protein specifically
            assets_pdb_paths = [
                LAMBO_FOLDX_ASSETS / "1zgo_A" / "wt_input_Repair.pdb"
            ]  # pick DsRed specifically.
        if n_allowed_observations > len(assets_pdb_paths):
            # if there is less data available than allowed, set this as starting number
            caller_info[STARTING_N] = len(assets_pdb_paths)
        observer = PoliBaseMlFlowObserver(TRACKING_URI)

        f, x0, y0 = RFPFoldXStabilityAndSASAProblemFactory().create(
            wildtype_pdb_path=assets_pdb_paths,
            batch_size=batch_size,
            seed=seed,
            parallelize=True,
            num_workers=num_workers,
            n_starting_points=n_allowed_observations,
        )

    # elif problem_name == "foldx_rfp_lambo":
    #     observer = PoliLamboLogger(TRACKING_URI)
    #     f, x0, y0 = RFPWrapperFactory().create(seed=seed)
    else:
        raise NotImplementedError

    observer.initialize_observer(f.info, caller_info, x0, y0, seed)

    f.set_observer(observer)

    return f, x0, y0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment specifications for running baselines on RFP."
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=0, help="Random seed for experiments."
    )
    parser.add_argument(
        "-m",
        "--max_evaluations",
        type=int,
        default=32,
        help="Optimization budget, number of possible observations.",
    )
    parser.add_argument(
        "-p",
        "--problem",
        type=str,
        choices=["foldx_rfp_lambo", "rfp_foldx_stability_and_sasa"],
        default="rfp_foldx_stability_and_sasa",
        help="Problem description as string key.",
    )
    parser.add_argument("-b", "--batch", type=int, default=16)
    parser.add_argument(
        "-n",
        "--number_observations",
        type=int,
        choices=AVAILABLE_SEQUENCES_N,
        default=AVAILABLE_SEQUENCES_N[0],
    )
    parser.add_argument(
        "-a",
        "--baseline-algorithm",
        type=str,
        choices=BASELINE_NAMES,
        default=BASELINE_NAMES[0],  # RandomMutation
    )
    args = parser.parse_args()

    problem_name = args.problem
    batch_size = args.batch
    seed = args.seed
    n_allowed_observations = args.number_observations
    n_iterations = args.max_evaluations

    # If the n_allowed_observations is 3, then the batch
    # size should be 3 as well
    if n_allowed_observations == 6:
        batch_size = 6

    n_mutations = 2

    # Adding some hardcoded values
    population_size = 16

    # Creating the black box
    caller_info = {
        BATCH_SIZE: batch_size,
        SEED: seed,
        STARTING_N: n_allowed_observations,
        MODEL: "BASELINE",
        ALGORITHM: f"{args.baseline_algorithm}_popsize_{population_size}_n_mutations_{n_mutations}",
    }
    f, x0, y0 = instantiate_black_box(
        problem_name,
        caller_info,
        n_allowed_observations=n_allowed_observations,
        batch_size=batch_size,
        seed=seed,
        num_workers=4,
    )

    # Making sure we're only using the allowed number of observations
    x0 = x0[:n_allowed_observations]
    y0 = y0[:n_allowed_observations]

    # Running the baseline
    baseline = instantiate_baseline(
        args.baseline_algorithm,
        f,
        x0,
        y0,
        n_mutations,
        batch_size,
        population_size,
    )

    saving_path = (
        PROJECT_ROOT_DIR
        / "results"
        / "baselines"
        / f"{problem_name}_{args.baseline_algorithm}_n_{n_allowed_observations}_seed_{seed}_b_{batch_size} / history.json"
    )
    saving_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.solve(
        max_iter=n_iterations,
        verbose=True,
        post_step_callbacks=[lambda solver: solver.save_history(saving_path)],
    )
