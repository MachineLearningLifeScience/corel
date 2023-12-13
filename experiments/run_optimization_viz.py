import os
from typing import List
from pathlib import Path
from itertools import product
import shutil
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from corel.observers import BLACKBOX, MIN_BLACKBOX, ABS_HYPER_VOLUME, REL_HYPER_VOLUME

# from . import TRACKING_URI

TRACKING_URI = "file:/Users/rcml/corel/results/slurm_mlruns/mlruns/"
METRIC_NAMES = [ABS_HYPER_VOLUME, REL_HYPER_VOLUME]


def optimization_figure(obs: np.ndarray, metric: str):
    plt.plot(obs)
    plt.title(metric)
    plt.show()


def get_metrics_from_run(runs, cache_path=None, metric_names=METRIC_NAMES) -> dict:
    metric_results = {m: [] for m in metric_names}
    for metric in metric_names:
        for r in MlflowClient().get_metric_history(runs.run_id.values[0], metric):
            metric_results[metric].append(r.value)
    if cache_path is not None:
        pass # TODO: write to ../results/cache/experiment_name_METRIC.pkl
    return metric_results


def remove_broken_mlflow_runs(tracking_uri) -> None:
    tracking_dir = tracking_uri.replace("file:", "")
    for experiment_dir in os.listdir(tracking_dir):
        experiment_dir_path = Path(tracking_dir) / experiment_dir
        for run_directory in os.listdir(experiment_dir_path):
            run_dir_path = experiment_dir_path / run_directory
            if not run_dir_path.is_dir():
                continue
            run_meta_file = run_dir_path / "meta.yaml"
            if os.stat(run_meta_file).st_size == 0:
                # if meta broken, remove complete run
                print(f"{run_meta_file} empty -> removing run directory")
                shutil.rmtree(run_dir_path)
    return


def correct_mlflow_artifact_paths(tracking_uri) -> None:
    raise NotImplementedError
    # TODO: for each run: read meta yaml, for artifact_uri replace home-dir with correct home dir
    return


def cleanup_mlflow_files(tracking_uri) -> None:
    remove_broken_mlflow_runs(tracking_uri)
    correct_mlflow_artifact_paths(tracking_uri)


def get_available_run_tag_combinations(runs) -> list:
    r_seeds = [r.data.tags["seed"] for r in runs]
    r_algos = [r.data.tags["ALGORITHM"] for r in runs]
    r_n0 = [r.data.tags["n_D0"] for r in runs]
    return list(zip(r_seeds, r_algos, r_n0))


def filter_run_results(experiment_combinations: list, runs: list) -> List[object]:
    # filter runs by required tags
    run_result = []
    existing_run_tag_combinations = get_available_run_tag_combinations(runs)
    # find first occurence in run list and load metric history
    for exp_tag_vals in experiment_combinations:
        if exp_tag_vals in existing_run_tag_combinations:
            results_idx = existing_run_tag_combinations.index(exp_tag_vals)
            run_result.append(runs[results_idx])
    return run_result


def get_algo_metric_history_from_run(mlf_client: object, run_results: list, algorithms: list, seeds=list):
    algo_metric_dict = {a: {s: {m: [] for m in METRIC_NAMES} for s in seeds} for a in algorithms}
    for metric, run in product(METRIC_NAMES, run_results):
        m_hist = mlf_client.get_metric_history(run.info.run_id, metric)
        m_values = [m.value for m in m_hist]
        algo = run.data.tags["ALGORITHM"]
        seed = run.data.tags["seed"]
        algo_metric_dict[algo][seed][metric] = m_values
    return metric_dict
    

if __name__ == "__main__":
    # cleanup_mlflow_files(TRACKING_URI)
    experiment_names = ["foldx_rfp_lambo", "foldx_stability_and_sasa"]
    RANDOM_SEEDS = ['0', '1', '5', '7', '13']
    ALGORITHMS = ["LAMBO", "COREL"] 
    starting_n = ['1', '50', '512']
    tag_keys = ["seed", "ALGORITHM", "n_D0"]
    experiment_combinations = product(RANDOM_SEEDS, ALGORITHMS, starting_n)

    # create one results df per experiment
    for experiment in experiment_names:
        mlf_client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
        tracked_experiment = mlf_client.get_experiment_by_name(experiment)
        # query MlFlow once for all runs
        runs = mlf_client.search_runs([tracked_experiment.experiment_id], run_view_type=ViewType.ACTIVE_ONLY) # TODO: only concluded runs!
        run_results = filter_run_results(experiment_combinations, runs)
        metric_dict = get_algo_metric_history_from_run(mlf_client, run_results, algorithms=ALGORITHMS, seeds=RANDOM_SEEDS)
        experiment_results_df = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k,v in metric_dict.items()}, axis=0)
        # TODO: aggregate by batch_size
        # TODO: visualize
