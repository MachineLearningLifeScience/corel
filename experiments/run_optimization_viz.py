import os
import shutil
from itertools import product
from pathlib import Path
from typing import List, Tuple

import lambo
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from botorch.utils.multi_objective import is_non_dominated
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from poli.core.util.proteins.mutations import \
    find_closest_wildtype_pdb_file_to_mutant
from torch import Tensor

from corel.observers import (ABS_HYPER_VOLUME, BLACKBOX, MIN_BLACKBOX,
                             REL_HYPER_VOLUME, UNNORMALIZED_HV)

TRACKING_URI = "file:/Users/rcml/corel/results/slurm_mlruns/mlruns/"
METRIC_DICT = {ABS_HYPER_VOLUME: "HV", 
                REL_HYPER_VOLUME: "rel. HV", 
                UNNORMALIZED_HV: "unnorm. HV",
                "blackbox_0": r"$f_0$", 
                "blackbox_1": r"$f_1$",
                "min_blackbox_0": r"$\min(f_0)$",
            }

rfp_label_markers_dict = {
    "DsRed.M1": "X", 
    "DsRed.T4": "o", 
    "mScarlet": "D", 
    "AdRed": "s", 
    "mRouge": "x", 
    "RFP630": "v"
}

algo_label_markers_dict = {
    "COREL": "X", 
    "LAMBO": "o", 
    "REFERENCE": "s",
    "RANDOM": "D", 
}

pareto_sequences_name_pdb_dict = {
            "DsRed.M1": "2VAD",
            "DsRed.T4": "2VAE",
            "mScarlet": "5LK4",
            "AdRed": "6AA7",
            "mRouge": "3NED",
            "RFP630": "3E5V",
        }

opt_colorscheme = ["#1E88E5", "#D81B60", "#FFC107", "#004D40", "#D81BAD", "#15CC80", "#FFB407", "#074D00"]

COLOR_DICT = dict(zip(algo_label_markers_dict.keys(), opt_colorscheme[:len(algo_label_markers_dict)]))

pareto_sequences_name_color_dict = dict(zip(pareto_sequences_name_pdb_dict.keys(), opt_colorscheme[:len(pareto_sequences_name_pdb_dict.keys())]))

figure_labels_kvp = {ABS_HYPER_VOLUME: "abs. hypervolume",
                    REL_HYPER_VOLUME: "rel. hypervolume",
                    "COREL": "CoRel",
                    "LAMBO": "LamBO",
                    "RandomMutation": "Random",
                    }


def obtain_pareto_front_idx_from_runs(df: pd.DataFrame, columns=["blackbox_0", "blackbox_1"]) -> np.ndarray:
    obs_tensor = Tensor([-df[columns[0]].values, -df[columns[1]].values]).T # NOTE: invert sign, conversion from minimization to original max problem
    pareto_vals = is_non_dominated(obs_tensor).numpy() # assumes maximization -> sign
    return np.where(pareto_vals)[0]


def obtain_starting_sequences_from_df_entry(df_entry: pd.DataFrame, artifact_name: str="sequences_observations.npz") -> Tuple[np.ndarray]:
    """
    For each pareto front entry obtain the artifact, load sequence and obtain selected sequence at step of the index.
    """
    results_path = Path(TRACKING_URI.replace("file:", ""))
    artifact_path = results_path / df_entry.run_uuid / "artifacts" / artifact_name
    entry_results = np.load(str(artifact_path.resolve())) 
    x0, y0 = entry_results["x0"], entry_results["y0"]
    return x0, y0


def obtain_sequences_from_df(df: pd.DataFrame, artifact_name: str="sequences_observations.npz") -> list:
    """
    For each pareto front entry obtain the artifact, load sequence and obtain selected sequence at step of the index.
    """
    # TODO: why is step count too high? BUG?
    pareto_seq_list = []
    for _, entry in df.iterrows(): # query the dataframe entries with the index that points to the pareto value
        results_path = Path(TRACKING_URI.replace("file:", ""))
        artifact_path = results_path / entry.run_uuid / "artifacts" / artifact_name
        entry_results = np.load(str(artifact_path.resolve())) 
        results_index = np.where(np.round(entry_results["y"][:,0], 5) == np.round(entry.blackbox_0, 5))[0]
        results_index_confirm = np.where(np.round(entry_results["y"][:,1], 5) == np.round(entry.blackbox_1, 5))[0]
        common_index = np.intersect1d(results_index, results_index_confirm)[0]
        pareto_seq_list.append(list(entry_results["x"][common_index]))
    return pareto_seq_list


def obtain_sequence_label_from_reference_file(candidate_sequences: list) -> list:
    reference_file = Path(lambo.__file__).parent.resolve() / "assets" / "fpbase" / "rfp_known_structures.csv"
    reference_pdb_path = Path(lambo.__file__).parent.resolve() / "assets" / "foldx"
    available_pdbs = os.listdir(reference_pdb_path)
    reference_df = pd.read_csv(reference_file)
    reference_pareto_pdb_filepaths = [reference_pdb_path / pdb / "wt_input_Repair.pdb" for pdb in available_pdbs if pdb.split("_")[0].upper() in pareto_sequences_name_pdb_dict.values()]
    reference_pareto_pdb_raw_filepaths = [reference_pdb_path / pdb / "wt_input.pdb" for pdb in available_pdbs if pdb.split("_")[0].upper() in pareto_sequences_name_pdb_dict.values()]
    labelled_sequences = []
    for i, seq in enumerate(candidate_sequences):
        try:
            pdb = find_closest_wildtype_pdb_file_to_mutant(wildtype_pdb_files=reference_pareto_pdb_filepaths, mutated_residue_string="".join(seq))
        except ValueError as e: # TODO: proposed sequences too long!
            # pdb = find_closest_wildtype_pdb_file_to_mutant(wildtype_pdb_files=reference_pareto_pdb_raw_filepaths, mutated_residue_string="".join(seq))
            print(f"Seq: {i} len={len(seq)} is unaligned -> unknown label")
            labelled_sequences.append("unknown")
            continue
        seq_closest_label = reference_df[reference_df["pdb_id"] == pdb._parts[-2].split("_")[0].upper()]["Name"].values[0]
        labelled_sequences.append(seq_closest_label)
    return labelled_sequences


def unpack_observations(df: pd.DataFrame, column: str) -> pd.DataFrame:
    unpacked_df = df.explode(column)
    unpacked_df = unpacked_df.reset_index(drop=True)
    # set indices of grouped algorithms
    unpacked_df["step"] = unpacked_df.groupby(["algorithm", "seed"]).cumcount()
    return unpacked_df


def pareto_front_figure(df: pd.DataFrame, columns=["blackbox_0", "blackbox_1"], title="") -> None:
    df_lst = []
    # TODO: load and add reference points x0, y0 from any of the runs!
    for i, col in enumerate(columns):
        if i == 0:
            columns = list(df.columns[:3]) + [col, "step", "run_uuid"]
        else:
            columns = [col]
        df_col = unpack_observations(df, column=col)
        df_lst.append(df_col[columns])
    df_combined = pd.concat(df_lst, axis=1, join="inner")
    figure_path = Path(__file__).parent.parent.resolve() / "results" / "figures"
    linestyles = [(1,1), (5,5), (2,2)]

    fig, ax = plt.subplots(figsize=(6, 6)) # subplot_kw={'aspect': 'equal'} TODO: make square
    # MAKE REFERENCE FRONT with points
    # load from experiment observations NOT from lambo reference files.
    x0, y0 = obtain_starting_sequences_from_df_entry(df_combined.iloc[0]) # starting pareto sequences should be consistent per experiment
    x0_labels = obtain_sequence_label_from_reference_file(x0)
    # these values are expected to be different when number of starting sequences are larger than pareto front!
    sns.lineplot(x=-y0[:,1], y=-y0[:,0],
            sort=True, estimator=None, dashes=linestyles[-1], linewidth=2., label="Start", color="black")
    
    for _y, label in zip(y0, x0_labels):
        sns.scatterplot(x=[-_y[1]], y=[-_y[0]], ax=ax, label=label, 
            marker=algo_label_markers_dict.get("REFERENCE"), 
            edgecolor="black", color=pareto_sequences_name_color_dict.get(label), s=72.)

    for i, algo in enumerate(df.algorithm.drop_duplicates()):
        if algo.startswith("Random"):
            continue
        # TODO: add reference values here
        subset_df = df_combined[df_combined["algorithm"] == algo]
        pareto_indices = obtain_pareto_front_idx_from_runs(subset_df)
        pareto_entries_df = subset_df.iloc[pareto_indices]
        sequence_files = obtain_sequences_from_df(pareto_entries_df)
        find_sequence_labels = obtain_sequence_label_from_reference_file(sequence_files)
        pareto_entries_df["labels"] = find_sequence_labels
        # palette = reds_palettes[i]
        pareto_entries_df["stability"] = - pareto_entries_df.blackbox_0.values # invert signs
        pareto_entries_df["SASA"] = - pareto_entries_df.blackbox_1.values

        sns.lineplot(pareto_entries_df, x="SASA", y="stability", 
                sort=True, estimator=None, dashes=linestyles[i], linewidth=2., label=figure_labels_kvp.get(algo.split("_")[0]), color="black")

        for label in pareto_sequences_name_pdb_dict.keys():
            plot_df = pareto_entries_df[pareto_entries_df["labels"] == label]
            sns.scatterplot(plot_df, x="SASA", y="stability", ax=ax, label=label, marker=algo_label_markers_dict.get(algo.split("_")[0]), 
            edgecolor="black", color=pareto_sequences_name_color_dict.get(label), s=72.)
    plt.grid(True, color="grey", linewidth=.15)
    plt.xlabel("SASA", fontsize=14)
    plt.xticks(rotation=45)
    plt.ylabel("Stability", fontsize=14)
    plt.tight_layout()
    batch_size = int(df_combined.iloc[0,0].split("_")[-1][1:])
    handles, labels = ax.get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax.legend(unique_legend)
    plt.savefig(f"{figure_path}/PARETO_OPT_experiment_{title.split()[0]}_batch{batch_size}.png")
    plt.savefig(f"{figure_path}/PARETO_OPT_experiment_{title.split()[0]}_batch{batch_size}.pdf")
    plt.show()


def cleanup_mlflow_files(tracking_uri: str) -> None:
    remove_broken_mlflow_runs(tracking_uri)
    correct_mlflow_artifact_paths(tracking_uri)


def enforce_equal_number_completed_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    For strict comparable plotting ensure that the number of seeds are the same!
    Return subset of the input df where the algorithm seeds are the minimal number of completed seeds for one of the algorithms.
    """
    print("Enforcing number of seeds completed...")
    unique_seeds = []
    for class_name in df.algorithm.unique():
        print(class_name)
        seeds = df[df["algorithm"] == class_name]["seed"].unique()
        unique_seeds.append(seeds)
    seed_numbers = [len(s) for s in unique_seeds]
    print(f"Completed seeds: {seed_numbers}")
    common_seeds_idx = seed_numbers.index(min(seed_numbers))
    out_df = df[df["seed"].isin(unique_seeds[common_seeds_idx])]
    return out_df


def optimization_line_figure(df: pd.DataFrame, metric: str, n_steps, title: str="", tick_every_batch: int=2, strict=False):
    if strict: # make sure all completed runs have the same number of seeds
        df = enforce_equal_number_completed_seeds(df)
    full_size_df = unpack_observations(df, column=metric)
    if n_steps:
        full_size_df = full_size_df[full_size_df.step <= n_steps]
    batch_size = int(full_size_df.iloc[0,0].split("_")[-1][1:])
    # HACK to overlay plots: point and lineplot treat x-axis differently, ensure categorical
    full_size_df["step_str"] = full_size_df.step.astype(str)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.lineplot(full_size_df, x="step_str", y=metric, hue="algorithm", ax=ax, palette=opt_colorscheme)
    batched_stats = full_size_df[full_size_df["step"] % batch_size == 0]
    sns.pointplot(batched_stats, x="step_str", y=metric, errorbar=("se", 1), capsize=.1, hue="algorithm", ax=ax, join="False", palette=opt_colorscheme)
    for line in ax.lines:
        line.set_markersize(3.)
        line.set_linewidth(1.)
    ax.set_xticks(np.arange(0, full_size_df["step"].max()+1, tick_every_batch*batch_size))
    ax.tick_params(axis="x", labelsize=14, rotation=45)
    ax.tick_params(axis="y", labelsize=14)
    plt.xlabel("steps", fontsize=16)
    plt.ylabel(METRIC_DICT.get(metric), fontsize=16)
    plt.title(title, fontsize=21)
    handles, labels = plt.gca().get_legend_handles_labels()
    updated_legend = dict(zip([figure_labels_kvp.get(label.split("_")[0]) for label in labels], handles))
    plt.legend(updated_legend.values(), updated_legend.keys())
    plt.subplots_adjust(top=0.91, right=0.978, left=0.15, bottom=0.21)
    figure_path = Path(__file__).parent.parent.resolve() / "results" / "figures" / "rfp"
    plt.savefig(f"{figure_path}/OPT_experiment_{metric.lower()}_{title.split()[0]}_batch{batch_size}.png")
    plt.savefig(f"{figure_path}/OPT_experiment_{metric.lower()}_{title.split()[0]}_batch{batch_size}.pdf")
    plt.show()
    

def optimization_figure(df: pd.DataFrame, metric=ABS_HYPER_VOLUME, title: str=""):
    # average over seeds:
    _ = unpack_observations(df, metric)
    algo_mean_vals = {a: {} for a in df.algorithm.unique()}
    for algo in df.algorithm.unique():
        values = df[df.algorithm == algo][metric].values
        if not all([len(v)==len(values[0]) for v in values]): # FOR DEVELOPMENT WHEN NOT ALL RUNS ARE CONCLUDED
            min_len = min([len(v) for v in values])
            values = [v[:min_len] for v in values]
        values = np.stack(values)
        algo_mean_vals[algo]["mean"] = np.mean(values, axis=0)
        algo_mean_vals[algo]["std"] = np.std(values, axis=0)

    for algo in df.algorithm.unique():
        plt.plot(algo_mean_vals[algo]["mean"], lw=4., linestyle="dashed", color=COLOR_DICT.get(algo), label=r"$\mu$"+f" {algo}")
        plt.fill_between(np.arange(len(algo_mean_vals[algo]["mean"])), algo_mean_vals[algo]["mean"] - 1.96*algo_mean_vals[algo]["std"], algo_mean_vals[algo]["mean"] + 1.96*algo_mean_vals[algo]["std"], alpha=0.25, 
            color=COLOR_DICT.get(algo), label=r"2$\sigma$"+f" {algo}")

    # plot individual trajectories
    for _, row in df.iterrows():
        plt.plot(row[metric], lw=4., color=COLOR_DICT.get(row.algorithm), label=row.algorithm, alpha=0.5)
    
    plt.xlabel(r"$f$ evaluation step", fontsize=18)
    plt.ylabel(METRIC_DICT.get(metric), fontsize=18)
    plt.title(title, fontsize=21)
    handles, labels = plt.gca().get_legend_handles_labels()
    l_h_dict = dict(zip(labels, handles))
    plt.legend(handles=list(l_h_dict.values()), labels=list(l_h_dict.keys()))
    figure_path = Path(__file__).parent.parent.resolve() / "results" / "figures"
    plt.savefig(f"{figure_path}/experiment_{metric.lower()}_{title.split()[0]}.png")
    plt.savefig(f"{figure_path}/experiment_{metric.lower()}_{title.split()[0]}.pdf")
    plt.show()


def get_metrics_from_run(runs, cache_path=None, metric_names=METRIC_DICT.keys()) -> dict:
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


def get_available_run_tag_combinations(runs) -> list:
    r_seeds = [r.data.tags["seed"] for r in runs]
    r_algos = [r.data.tags["ALGORITHM"].split("_")[0] for r in runs] # grep for the algoname if longer string concatenation
    r_n0 = [r.data.tags["n_D0"] for r in runs]
    r_batch = [r.data.tags["BATCH_SIZE"] for r in runs]
    return list(zip(r_seeds, r_algos, r_n0, r_batch))


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


def get_algo_metric_history_from_run(mlf_client: object, run_results: list, algorithms: list, seeds=list, batch_sizes=list, 
                            starting_n=list, metric_names=METRIC_DICT.keys()):
    algos = [f"{a}_b{b}" for a, b in product(algorithms, batch_sizes)]
    algo_metric_dict = {a: {s: {**{"starting_N": n for n in starting_n}, 
                                **{m: [] for m in metric_names}} 
                        for s in seeds} 
                    for a in algos}
    for metric, run in product(metric_names, run_results):
        m_hist = mlf_client.get_metric_history(run.info.run_id, metric)
        m_values = [m.value for m in m_hist]
        algo = f"{run.data.tags['ALGORITHM'].split('_')[0]}_b{run.data.tags['BATCH_SIZE']}"
        seed = run.data.tags["seed"]
        algo_metric_dict[algo][seed]["starting_N"] = run.data.tags["n_D0"]
        algo_metric_dict[algo][seed][metric] = m_values
        algo_metric_dict[algo][seed]["run_uuid"] = run.info.experiment_id + "/" + run.info.run_uuid  # required for later artifact loading
    return algo_metric_dict


def load_viz_rfp_experiments(exp_name: str="rfp_foldx_stability_and_sasa",
            algorithms: List[str]=["LAMBO", "COREL", "RandomMutation"],
            seeds: List[str]=["0", "1", "5", "7", "13", "42", "17", "23", "42", "71", "123", "29", "37", "73"], 
            starting_n: List[str]=['6', '50'],
            batch_size: List[str] = ['6', '16'],
            tag_keys: List[str]=["seed", "ALGORITHM", "n_D0"],
            finished_only = True, # False ## DEBUG
            strict=True,
            n_steps: int=180,
            pareto_fig=False,
            ):
    experiment_combinations = product(seeds, algorithms, starting_n, batch_size)
    mlf_client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    tracked_experiment = mlf_client.get_experiment_by_name(exp_name)
    # query MlFlow once for all runs
    runs = mlf_client.search_runs([tracked_experiment.experiment_id], run_view_type=ViewType.ACTIVE_ONLY)
    if finished_only:
        runs = [r for r in runs if r.info.status == "FINISHED"]
    run_results = filter_run_results(experiment_combinations, runs)
    metric_dict = get_algo_metric_history_from_run(mlf_client, run_results, algorithms=algorithms, seeds=seeds, batch_sizes=batch_size, starting_n=starting_n)
    experiment_results_df = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k,v in metric_dict.items()}, axis=0)
    experiment_results_df = experiment_results_df.reset_index().rename(columns={"level_0": "algorithm", "level_1": "seed"})
    experiment_combinations = product(seeds, algorithms, starting_n, batch_size)
    mlf_client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    tracked_experiment = mlf_client.get_experiment_by_name(exp_name)
    # query MlFlow once for all runs
    runs = mlf_client.search_runs([tracked_experiment.experiment_id], run_view_type=ViewType.ACTIVE_ONLY)
    if finished_only:
        runs = [r for r in runs if r.info.status == "FINISHED"]
    run_results = filter_run_results(experiment_combinations, runs)
    metric_dict = get_algo_metric_history_from_run(mlf_client, run_results, algorithms=algorithms, seeds=seeds, batch_sizes=batch_size, starting_n=starting_n)
    experiment_results_df = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k,v in metric_dict.items()}, axis=0)
    experiment_results_df = experiment_results_df.reset_index().rename(columns={"level_0": "algorithm", "level_1": "seed"})
    cold_experiments = experiment_results_df[(experiment_results_df.algorithm.str.endswith("_b6")) & (experiment_results_df.starting_N.astype(int) == 6)]
    warm_experiments = experiment_results_df[(experiment_results_df.algorithm.str.endswith("_b16")) & (experiment_results_df.starting_N.astype(int) == 50)]
    ref_experiments = experiment_results_df[(experiment_results_df.algorithm.str.endswith("_b16")) & (experiment_results_df.starting_N.astype(int) >= 500)]
    ## OPTIMIZATION FIGURES
    for metric in METRIC_DICT.keys():
        if exp_name != "foldx_rfp_lambo":
            optimization_line_figure(cold_experiments[["algorithm", "seed", "starting_N", metric]], metric=metric, title="cold HV optimization N=6", strict=strict, n_steps=n_steps)
            optimization_line_figure(warm_experiments[["algorithm", "seed", "starting_N", metric]], metric=metric, title="warm HV optimization N=50", strict=strict, n_steps=None)
        else:
            optimization_line_figure(ref_experiments[["algorithm", "seed", "starting_N", metric]], metric=metric, title="ref. HV optimization N=512", strict=strict, n_steps=n_steps)
    if pareto_fig:
        ## PARETO FIGURES
        pareto_front_figure(cold_experiments[["algorithm", "seed", "starting_N", "run_uuid", "blackbox_0", "blackbox_1"]], title="cold optimization\nN=6")
        pareto_front_figure(warm_experiments[["algorithm", "seed", "starting_N", "run_uuid", "blackbox_0", "blackbox_1"]], title="cold optimization\nN=50")


def make_performance_and_regret_figure(df: pd.DataFrame, target: str="blackbox_0") -> None:
    unpacked_df = unpack_observations(df, column=target)[["algorithm", "seed", "starting_N", "step", target]]
    unpacked_df_min = unpack_observations(df, column="min_"+target)[["algorithm", "seed", "starting_N", "step", "min_"+target]]
    batch_size = unpacked_df.iloc[0,0].split("_")[-1]
    def filter_unique_counts(group):
        return group["step"].nunique() == unpacked_df.step.max()+1
    unpacked_df = unpacked_df.groupby(["algorithm", "seed"]).filter(filter_unique_counts)
    unpacked_df_min = unpacked_df_min.groupby(["algorithm", "seed"]).filter(filter_unique_counts)
    min_number_seeds = unpacked_df.groupby(["algorithm"])["seed"].nunique().min()
    print(f"Minimal number of seeds: {min_number_seeds}")
    subselected_algo_dfs = []
    subselected_absmin_algo_dfs = []
    for algo in unpacked_df.algorithm.unique(): # filter by minimal amount of overlapping seeds
        algo_df = unpacked_df[unpacked_df.algorithm==algo]
        algo_min_df = unpacked_df_min[unpacked_df_min.algorithm==algo]
        min_seeds_for_algo = algo_df.seed.unique()[:min_number_seeds]
        min_seeds_for_algo_min = algo_min_df.seed.unique()[:min_number_seeds]
        subselected_df = algo_df[algo_df.seed.isin(min_seeds_for_algo)]
        subselected_min_df = algo_min_df[algo_min_df.seed.isin(min_seeds_for_algo_min)]
        subselected_algo_dfs.append(subselected_df)
        subselected_absmin_algo_dfs.append(subselected_min_df)
    filtered_results_df = pd.concat(subselected_algo_dfs)
    filtered_min_results_df = pd.concat(subselected_absmin_algo_dfs)
    filtered_results_df["opt_target"] = -filtered_results_df[target].values
    filtered_min_results_df["opt_target"] = -filtered_min_results_df["min_"+target].values
    # step-wise observations figure
    sns.lineplot(filtered_results_df, x="step", y="opt_target", hue="algorithm", 
                hue_order=["COREL_b3", "RandomMutation_b3"], palette=opt_colorscheme)
    plt.ylabel(METRIC_DICT.get(target), fontsize=14)
    plt.xlabel("steps", fontsize=14)
    figure_path = Path(__file__).parent.parent.resolve() / "results" / "figures" / "gfp"
    plt.savefig(f"{figure_path}/observations_{target}_batch{batch_size}.png")
    plt.savefig(f"{figure_path}/observations_{target}_batch{batch_size}.pdf")
    plt.show()
    # abs. min observations figure
    sns.lineplot(filtered_min_results_df, x="step", y="opt_target", hue="algorithm", 
                hue_order=["COREL_b3", "RandomMutation_b3"], palette=opt_colorscheme)
    plt.ylabel(METRIC_DICT.get(target), fontsize=14)
    plt.xlabel("steps", fontsize=14)
    plt.savefig(f"{figure_path}/BEST_observations_{target}_batch{batch_size}.png")
    plt.savefig(f"{figure_path}/BEST_observations_{target}_batch{batch_size}.pdf")
    plt.show()

    # # TODO: compute cumulative regret for each step
    # # TODO: load global minimum from GFP data
    # # TODO: subtract absmin plot from GFP oracle best value
    # stepwise_regret_df = None
    # sns.lineplot(stepwise_regret_df, x="step", y="cml_regret")
    # plt.show()


def load_viz_gfp_experiments(
    exp_name: str = "gfp_cbas_gp",
    seeds: List[str] = ["0", "1", "5", "7", "13", "42", "17", "23", "42", "71", "123", "29", "37", "73"],
    algorithms: List[str] = ["COREL", "RandomMutation"],
    starting_n: List[str] = ['3', '50'],
    batch_size: List[str] = ['3', '16'],
    tag_keys: List[str] = ["seed", "ALGORITHM", "n_D0"],
    finished_only: bool = False, ## DEBUG # NOTE: not all GFP experiments are terminated in Mlflow
    strict: bool = True, # False ## DEBUG # NOTE: enforce equal length of steps
):
    experiment_combinations = product(seeds, algorithms, starting_n, batch_size)
    mlf_client = mlflow.tracking.MlflowClient(tracking_uri=TRACKING_URI)
    tracked_experiment = mlf_client.get_experiment_by_name(exp_name)
    # query MlFlow once for all runs
    runs = mlf_client.search_runs([tracked_experiment.experiment_id], run_view_type=ViewType.ACTIVE_ONLY)
    if finished_only:
        runs = [r for r in runs if r.info.status == "FINISHED"]
    run_results = filter_run_results(experiment_combinations, runs)
    metric_dict = get_algo_metric_history_from_run(mlf_client, run_results, algorithms=algorithms, seeds=seeds, batch_sizes=batch_size, starting_n=starting_n)
    experiment_results_df = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k,v in metric_dict.items()}, axis=0)
    experiment_results_df = experiment_results_df.reset_index().rename(columns={"level_0": "algorithm", "level_1": "seed"})
    experiments_df = experiment_results_df[(experiment_results_df.algorithm.str.endswith("_b3")) & (experiment_results_df.starting_N.astype(int) == 3)] # we care only about N=3 experiments
    make_performance_and_regret_figure(experiments_df)


if __name__ == "__main__":
    ## LOAD AND VISUALIZE RFP EXPERIMENTS
    # RFP base experiments
    # load_viz_rfp_experiments(pareto_fig=True)
    # # RFP reference experiments
    # load_viz_rfp_experiments(exp_name="foldx_rfp_lambo", starting_n=["512"], finished_only=False)
    ## LOAD AND VISUALIZE GFP EXPERIMENTS
    load_viz_gfp_experiments()


