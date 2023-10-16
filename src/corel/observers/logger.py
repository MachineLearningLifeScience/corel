import logging
import warnings
from uuid import uuid4
import mlflow
import os
from pathlib import Path

from corel.observers import SEQUENCE, SEED


EXPERIMENT_PATH = Path(Path(__file__).parents[3], "results/mlruns")

tracking_uri = EXPERIMENT_PATH.as_posix()
logging.info("Tracking results in " + tracking_uri)
mlflow.set_tracking_uri(tracking_uri)


def initialize_logger(problem_setup_info, caller_info, seed, run_id=None):
    experiment = mlflow.set_experiment(experiment_name=problem_setup_info.get_problem_name())
    if "ALGORITHM" in caller_info.keys():
        run_name = f"{caller_info.get('ALGORITHM')}_b{caller_info.get('BATCH_SIZE')}_n{caller_info.get('n_D0')}_s{caller_info.get('seed')}_{str(uuid4().int)[:6]}"
    else:
        run_name = None
    if run_id is None:
        run = mlflow.start_run(
            run_name=run_name,
            experiment_id=experiment.experiment_id
        )
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