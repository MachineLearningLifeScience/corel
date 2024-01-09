"""This script runs the baselines on the different objectives.

Baselines to run:
- Random hill climbing
- NSGA-II (for the multi-objective problems)

Objective functions to optimize:
- "gfp_cbas_gp" ( )
- "gfp_cbas_elbo" ( )
- "foldx_rfp_lambo" ( )
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
from typing import Tuple
from pathlib import Path

import numpy as np

from pymoo.core.infill import InfillCriterion
from pymoo.core.mutation import Mutation

from poli.core.problem_setup_information import ProblemSetupInformation

from poli.objective_repository import FoldXStabilityAndSASAProblemFactory

from poli_baselines.solvers import (
    DiscreteNSGAII,
)

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


if __name__ == "__main__":
    # TODO: add argparse.
    batch = 1
    seed = 0
    n_allowed_observations = 2

    observer = PoliLamboLogger(TRACKING_URI)
    caller_info = {
        BATCH_SIZE: batch,
        SEED: seed,
        STARTING_N: n_allowed_observations,
        MODEL: "BASELINE",
        ALGORITHM: "NSGAII",
    }

    wildtype_pdb_paths = list(LAMBO_FOLDX_ASSETS.glob("*/wt_input_Repair.pdb"))[
        :n_allowed_observations
    ]

    problem_factory = FoldXStabilityAndSASAProblemFactory()

    f, x0, y0 = problem_factory.create(
        wildtype_pdb_path=wildtype_pdb_paths,
        batch_size=batch,
    )

    x0 = x0[:n_allowed_observations]
    y0 = y0[:n_allowed_observations]

    # x0, y0 = prepare_data_for_experiment(x0, y0, n_allowed_observations, problem_info)

    baseline = DiscreteNSGAII(
        black_box=f,
        x0=x0,
        y0=y0,
        population_size=x0.shape[0],
    )

    baseline.solve(max_iter=10)
