"""This script runs the baselines on the different objectives.

Baselines to run:
- Random hill climbing
- NSGA-II (for the multi-objective problems)

Objective functions to optimize:
- "gfp_cbas_gp" ( )
- "gfp_cbas_elbo" ( )
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
from typing import Tuple, Literal
from pathlib import Path
import warnings
import argparse

warnings.filterwarnings("ignore", module="Bio")

import numpy as np

from poli.objective_repository import RFPWrapperFactory
from poli.core.problem_setup_information import ProblemSetupInformation

from poli.objective_repository import FoldXStabilityAndSASAProblemFactory

from poli_baselines.solvers import (
    DiscreteNSGAII,
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

AVAILABLE_SEQUENCES_N = [3, 16, 50, 512]


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
    assets_pdb_paths = list(LAMBO_FOLDX_ASSETS.glob("*/wt_input_Repair.pdb"))

    # This is mostly taken from run_cold_warm_start_experiments_rfp_bo.py
    if problem_name == "foldx_stability_and_sasa":
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

        f, x0, y0 = FoldXStabilityAndSASAProblemFactory().create(
            wildtype_pdb_path=assets_pdb_paths,
            batch_size=4,
            seed=seed,
            parallelize=True,
            num_workers=4,
        )

    elif problem_name == "foldx_rfp_lambo":
        observer = PoliLamboLogger(TRACKING_URI)
        f, x0, y0 = RFPWrapperFactory().create(seed=seed)
    else:
        raise NotImplementedError

    observer.initialize_observer(f.info, caller_info, x0, y0, seed)

    f.set_observer(observer)

    return f, x0, y0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment specifications for running NSGA-II."
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
        choices=["foldx_rfp_lambo", "foldx_stability_and_sasa"],
        default="foldx_rfp_lambo",
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
    population_size = 10
    n_iterations = 10
    n_mutations = 5

    # Creating the black box
    caller_info = {
        BATCH_SIZE: batch,
        SEED: seed,
        STARTING_N: n_allowed_observations,
        MODEL: "BASELINE",
        ALGORITHM: f"NSGAII_popsize_{population_size}_n_mutations_{n_mutations}",
    }
    f, x0, y0 = instantiate_black_box(problem_name, caller_info)

    # Making sure we're only using the allowed number of observations
    x0 = x0[:n_allowed_observations]
    y0 = y0[:n_allowed_observations]

    # Running the baseline
    baseline = DiscreteNSGAII(
        black_box=f if args.problem == "foldx_stability_and_sasa" else -f,
        x0=x0,
        y0=y0,
        population_size=population_size,
        num_mutations=n_mutations,
    )

    saving_path = (
        PROJECT_ROOT_DIR
        / "results"
        / "baselines"
        / f"{problem_name}_n_{n_allowed_observations}_seed_{seed} / history.json"
    )
    saving_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.solve(
        max_iter=n_iterations,
        verbose=True,
        post_step_callbacks=[lambda solver: solver.save_history(saving_path)],
    )
