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
from poli.core.abstract_black_box import AbstractBlackBox
from poli.objective_repository.gfp_cbas.register import GFPCBasProblemFactory

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
BASELINE_NAMES = ["RandomMutation"]

PROJECT_ROOT_DIR = Path(__file__).parent.parent.resolve()
TRACKING_URI = f"file:/{PROJECT_ROOT_DIR}/results/mlruns"


def instantiate_black_box(
    problem_name: Literal["foldx_stability_and_sasa", "foldx_rfp_lambo"],
    caller_info: dict,
):
    observer = None
    observer = PoliBaseMlFlowObserver(TRACKING_URI)
    info("Invoking objective factory create")
    # f, x0, y0 = GFPCBasProblemFactory().create(
    #     seed=seed,
    #     caller_info=caller_info,
    #     batch_size=batch_size,  # determines return shape of y0
    #     observer=observer,
    #     force_register=True,
    #     parallelize=False,
    #     problem_type=problem_name.split("_")[-1],
    # )
    problem_info, f, x0, y0, run_info = objective_factory.create(
        name=problem_name,
        seed=seed,
        caller_info=caller_info,
        batch_size=batch_size,  # determines return shape of y0
        observer=observer,
        force_register=True,
        parallelize=False,
        problem_type=problem_name.split("_")[-1],
        n_starting_points=n_allowed_observations,
    )

    return f, x0, y0


def instantiate_baseline(
    baseline_name: Literal["RandomMutation", "FixedLengthGeneticAlgorithm"],
    f: AbstractBlackBox,
    x0: np.ndarray,
    y0: np.ndarray,
    n_mutations: int,
    batch_size: int,
    prob_mutations: float = 0.5,
):
    if baseline_name == "RandomMutation":
        baseline = RandomMutation(
            black_box=-f,
            x0=x0,
            y0=-y0,
            n_mutations=n_mutations,
            top_k=batch_size,
            batch_size=batch_size,
            greedy=True,
        )
    elif baseline_name == "FixedLengthGeneticAlgorithm":
        baseline = FixedLengthGeneticAlgorithm(
            black_box=-f,
            x0=x0,
            y0=-y0,
            population_size=population_size,
            prob_of_mutation=prob_mutations,
            minimize=False,
        )
    else:
        raise ValueError(f"Unknown baseline {baseline_name}")

    return baseline


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment specifications for running random mutations."
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=0, help="Random seed for experiments."
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
    parser.add_argument("-b", "--batch", type=int, default=16)
    parser.add_argument(
        "-a",
        "--baseline-algorithm",
        type=str,
        choices=BASELINE_NAMES,
        default=BASELINE_NAMES[0],  # RandomMutation
    )
    parser.add_argument(
        "-n",
        "--number_observations",
        type=int,
        choices=AVAILABLE_SEQUENCES_N,
        default=AVAILABLE_SEQUENCES_N[0],
    )
    args = parser.parse_args()

    problem_name = args.problem
    batch_size = args.batch
    seed = args.seed
    n_allowed_observations = args.number_observations

    # If the n_allowed_observations is 3, then the batch
    # size should be 3 as well
    if n_allowed_observations == 3:
        batch_size = 3

    # Adding some hardcoded values for the baselines
    population_size = batch_size
    n_mutations = 2

    # Creating the black box
    set_seeds(seed)
    caller_info = {
        BATCH_SIZE: batch_size,
        SEED: seed,
        STARTING_N: n_allowed_observations,
        MODEL: "BASELINE",
        ALGORITHM: f"{args.baseline_algorithm}_popsize_{population_size}_n_mutations_{n_mutations}",
    }
    f, x0, y0 = instantiate_black_box(problem_name, caller_info)

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
    )

    saving_path = (
        PROJECT_ROOT_DIR
        / "results"
        / "baselines"
        / f"{problem_name}_{args.baseline_algorithm}_n_{n_allowed_observations}_seed_{seed}_batch_{batch_size} / history.json"
    )
    saving_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.solve(
        max_iter=args.max_evaluations,
        verbose=True,
        post_step_callbacks=[lambda solver: solver.save_history(saving_path)],
    )

    f.terminate()
