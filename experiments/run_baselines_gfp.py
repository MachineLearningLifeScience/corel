"""This script runs the baselines on the different objectives.

Baselines to run:
- Random mutations
- GA

Objective functions to optimize:
- "gfp_cbas_gp",
- "gfp_cbas_elbo"

To run this script, you'll need to:

1. Set up poli.
    - Clone poli and install it:
      pip install git+https://github.com/MachineLearningLifeScience/poli.git@dev

2. Set up poli-baselines.
    - Clone poli-baselines and install it:
      pip install git+https://github.com/MachineLearningLifeScience/poli-baselines
"""
from typing import Tuple, Literal
from pathlib import Path
import warnings
import argparse
from logging import info

warnings.filterwarnings("ignore", module="Bio")

import numpy as np

from poli import objective_factory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli_baselines.solvers import FixedLengthGeneticAlgorithm, RandomMutation

from corel.observers.poli_base_logger import PoliBaseMlFlowObserver
from corel.observers.poli_lambo_logger import PoliLamboLogger
from corel.util.constants import (
    ALGORITHM,
    BATCH_SIZE,
    SEED,
    STARTING_N,
    MODEL,
)
from corel.util.util import transform_string_sequences_to_string_arrays, set_seeds

AVAILABLE_SEQUENCES_N = [3, 16, 50, 512]

PROBLEM_NAMES = ["gfp_cbas_gp", "gfp_cbas_elbo"]

PROJECT_ROOT_DIR = Path(__file__).parent.parent.resolve()
TRACKING_URI = f"file:/{PROJECT_ROOT_DIR}/results/mlruns"


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
    problem_name: Literal["foldx_stability_and_sasa", "foldx_rfp_lambo"],
    caller_info: dict,
):
    observer = None
    observer = PoliBaseMlFlowObserver(TRACKING_URI)
    info("Invoking objective factory create")
    problem_info, f, x0, y0, run_info = objective_factory.create(
        name=problem_name,
        seed=seed,
        caller_info=caller_info,
        batch_size=batch,  # determines return shape of y0
        observer=observer,
        force_register=True,
        parallelize=False,
        problem_type=problem_name.split("_")[-1],
    )

    return f, x0, y0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment specifications for running NSGA-II."
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=1, help="Random seed for experiments."
    )
    parser.add_argument(
        "-m",
        "--max_evaluations",
        type=int,
        default=100,
        help="Optimization budget, number of possible observations.",
    )
    parser.add_argument(
        "-p",
        "--problem",
        type=str,
        choices=PROBLEM_NAMES,
        default=PROBLEM_NAMES[0],
        help="Problem description as string key.",
    )
    parser.add_argument("-b", "--batch", type=int, default=None)
    parser.add_argument(
        "-n",
        "--number_observations",
        type=int,
        choices=AVAILABLE_SEQUENCES_N,
        default=AVAILABLE_SEQUENCES_N[-1],
    )
    args = parser.parse_args()

    problem_name = args.problem
    batch = args.batch
    seed = args.seed
    n_allowed_observations = args.number_observations

    # Adding some hardcoded values for NSGA-II
    population_size = 100
    n_iterations = 500
    prob_mutations = 100 / 100
    n_mutations = 1

    # Creating the black box
    set_seeds(seed)
    caller_info = {
        BATCH_SIZE: batch,
        SEED: seed,
        STARTING_N: n_allowed_observations,
        MODEL: "BASELINE",
        ALGORITHM: f"RANDOM_popsize_{population_size}_n_mutations_{n_mutations}",
    }
    f, x0, y0 = instantiate_black_box(problem_name, caller_info)

    # Making sure we're only using the allowed number of observations
    x0 = x0[:n_allowed_observations]
    y0 = y0[:n_allowed_observations]

    # Running the baseline
    # baseline = FixedLengthGeneticAlgorithm(
    #     black_box=f,
    #     x0=x0,
    #     y0=y0,
    #     population_size=population_size,
    #     prob_of_mutation=prob_mutations,
    #     minimize=False,
    # )
    baseline = RandomMutation(
        black_box=-f,
        x0=x0,
        y0=-y0,
        n_mutations=n_mutations,
    )

    saving_path = (
        PROJECT_ROOT_DIR
        / "results"
        / "baselines"
        / f"{problem_name}_RandomMutation_n_{n_allowed_observations}_seed_{seed} / history.json"
    )
    saving_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.solve(
        max_iter=n_iterations,
        verbose=True,
        post_step_callbacks=[lambda solver: solver.save_history(saving_path)],
    )

    f.terminate()
