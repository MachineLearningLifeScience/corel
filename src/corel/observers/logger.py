import logging
import warnings

import mlflow
import os
from pathlib import Path

from corel.observers.constants import SEQUENCE, SEED


DEBUG = False
EXPERIMENT_PATH = Path(Path(__file__).parents[2], "results/mlruns")

tracking_uri = EXPERIMENT_PATH
if DEBUG:
    logging.fatal("Tracking results in " + tracking_uri)
mlflow.set_tracking_uri(tracking_uri)


def initialize_logger(problem_setup_info, caller_info, seed, run_id=None):
    mlflow.set_experiment(experiment_name=problem_setup_info.get_problem_name())
    if run_id is None:
        run = mlflow.start_run()
        mlflow.set_tags(caller_info)
        mlflow.set_tag(SEED, str(seed))
    else:
        run = mlflow.start_run(run_id=run_id)
    return run.info.run_id


def log(d: dict, step: int, verbose=True):
    if verbose:
        for k in d.keys():
            print('\033[35m' + f"{k}: {d[k]}" + '\033[0m')
    mlflow.log_metrics(d, step=step)


def log_vector(key, vec, step=None, verbose=False):
    d = dict()
    for i in range(len(vec)):
        d[key + str(i)] = vec[i]
    log(d, step=step, verbose=verbose)


def log_sequence(seq, step=None, verbose=False):
    warnings.warn("not logging sequences currently")
    return
    log_vector(SEQUENCE, seq, step=step, verbose=verbose)


def finish():
    mlflow.end_run()