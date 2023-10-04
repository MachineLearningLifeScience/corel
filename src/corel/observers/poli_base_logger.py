__author__ = 'Richard Michael'

"""
This module extends an observer using mlflow for the optimization setting computing MT observations in botorch.

This requires an environment with mlflow , pytorch and botorch installed.
We recommend the environment with which lambo results have been collected for reproducability.

To check its results, you will need to start a ui:

    mlflow ui --backend-store-uri ./mlruns
"""

from pathlib import Path
import logging
from typing import Tuple
import mlflow
import numpy as np
import torch
from botorch.utils.multi_objective import infer_reference_point, Hypervolume, is_non_dominated

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.registry import set_observer
from poli.core.util.abstract_observer import AbstractObserver

import corel
from corel.observers import HD_PREV, HD_WT, HD_MIN, SEQUENCE, BLACKBOX, MIN_BLACKBOX, ABS_HYPER_VOLUME, REL_HYPER_VOLUME
from corel.observers.logger import log, initialize_logger, finish, log_sequence

# NOTE: this here is a very particular Normalizer (presented in the LamBO work)
from lambo.optimizers.pymoo import Normalizer


STARTING_N = "n_D0"
ALGORITHM = "ALGORITHM"
BATCH_SIZE = "BATCH_SIZE"


def get_pareto_reference_point(y0: np.ndarray) -> Tuple[torch.Tensor, Normalizer]:
    """
    This is a pareto reference implementaiton from discrete-bo.
    TODO: refactor into Lambo Poli logger.
    """
    target_min = y0.min(axis=0).copy()
    target_range = y0.max(axis=0).copy() - target_min
    hypercube_transform = Normalizer(
        loc=target_min + 0.5 * target_range,
        scale=target_range / 2.,
    )
    transform = hypercube_transform
    idx = is_non_dominated(-torch.tensor(y0))
    norm_pareto_targets = hypercube_transform(y0[idx.numpy(), ...])
    normed_ref_point = -infer_reference_point(-torch.tensor(norm_pareto_targets)).numpy()
    return normed_ref_point, transform


class PoliBaseMlFlowObserver(AbstractObserver):
    def __init__(self, tracking_uri: Path=None) -> None:
        self.step = 0
        self.tracking_uri = "file:/Users/rcml/corel/results/mlruns/" if tracking_uri is None else tracking_uri #Path(corel.__file__).parent.parent.parent.resolve() / "results" # tracking_uri
        logging.info(f"Setting MlFlow tracking = {self.tracking_uri}")
        self.initial_values = []
        self.initial_sequences = []
        self.values = []
        self.sequences = []
        self.additional_metrics = {}
        self.info = None

        self.transformed_pareto_volume_ref_point = None
        super().__init__()

    def initialize_observer(
        self,
        problem_setup_info: ProblemSetupInformation,
        caller_info: object,
        x0: np.ndarray,
        y0: np.ndarray,
        seed: int,
    ) -> None:
        
        self.info = problem_setup_info
        self.run_id = initialize_logger(problem_setup_info, caller_info, seed)

        mlflow.log_params(
            {
                "name": problem_setup_info.name,
                "max_sequence_length": problem_setup_info.max_sequence_length,
                "alphabet": problem_setup_info.alphabet,
            }
        )

        # due to Mlflow constraints only log initial 10 values:
        for i, _x in enumerate(x0[:10]):
            mlflow.log_param(f"x0_{i}", "".join(_x)) # can only write string for one X, numpy array/string length length exceeds limit
        mlflow.log_param("y0", y0[:10])
        mlflow.log_param("seed", seed)
        mlflow.log_param(ALGORITHM, caller_info.get(ALGORITHM))
        mlflow.log_param(STARTING_N, caller_info.get(STARTING_N))
        mlflow.log_param(BATCH_SIZE, caller_info.get(BATCH_SIZE))
        # compute and set initial front:
        self._add_initial_observations(x0, y0)
        # for completeness log and write to np array later
        self.initial_values.append(y0)
        self.initial_sequences.append(x0)

    def __add_to_additional_metrics(self, k: str, v: list) -> None:
        if k not in self.additional_metrics:
            self.additional_metrics[k] = []
        self.additional_metrics[k].append(v)

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        if context is not None: # log model parameters if context is given
            for key, value in context.items():
                if isinstance(value, list) or isinstance(value, np.array):
                    if len(np.atleast_1d(value[0])) > 1: # if metric is a vector/matrix, add all to artifacts
                        self.__add_to_additional_metrics(key, value)
                        continue
                    for idx, val in enumerate(value): # if sufficient size, add to metrics
                        mlflow.log_metric(f"{key}_{idx}", val, step=self.step)
                else:
                    mlflow.log_metric(key, value, step=self.step)
        self.values.append(y)
        self.sequences.append(x)
        
        log({f"{BLACKBOX}_{i}": float(y[:, i]) for i in range(y.shape[1])}, step=self.step)
        ymat = np.array(self.values)
        mins = np.min(ymat, axis=0) 
        assert(mins.shape[1] == y.shape[1]), "Mismatch min dimensions and observations!"
        log({f"{MIN_BLACKBOX}_{i}": float(mins[:,i]) for i in range(y.shape[1])}, step=self.step)
        if self.transformed_pareto_volume_ref_point is not None and y.shape[1] > 1:
            new_volume = self._compute_hyper_volume(ymat)
            log({ABS_HYPER_VOLUME: new_volume, 
                REL_HYPER_VOLUME: new_volume / self.initial_pareto_front_volume},
                step=self.step)
        self.step += 1

    def _add_initial_observations(self, x0, y0):
        self.step = -len(x0)
        for i in range(y0.shape[0]):
            # each element in the vector is a dedicated observation
            self.observe(x0[i:i+1, ...], y0[i:i + 1, ...])
        # the assertion assumes that the step increase is executed in the beginning of #observe
        assert(self.step == 0)
        if y0.shape[1] > 1:
            transformed_pareto_volume_ref_point, self.transform = get_pareto_reference_point(y0)
            self.transformed_pareto_volume_ref_point = torch.Tensor(transformed_pareto_volume_ref_point)
            self.initial_pareto_front_volume = self._compute_hyper_volume(y0)
            log({ABS_HYPER_VOLUME: self.initial_pareto_front_volume}, step=self.step)
        
    def _compute_hyper_volume(self, all_y: np.ndarray) -> float:
        """
        Compute Hypervolume as a maximization problem.
        Take transformed pareto V reference point, compute w.r.t. normalized pareto targets.
        NOTE: Reference BoTorch implementation (LaMBO) used.
        NOTE: y observations are not negated for normalized pareto targets.
        """
        tymat = torch.Tensor(all_y)
        idx = is_non_dominated(-tymat).numpy()
        norm_pareto_targets = self.transform(all_y[idx, ...])
        # this implementation of volume computation assumes maximization
        return Hypervolume(-self.transformed_pareto_volume_ref_point).compute(torch.tensor(-norm_pareto_targets))

    def finish(self) -> None:
        _seqs = ["".join(list(s[0])) for s in self.sequences if s.dtype == "<U1"]
        sequences = np.array(_seqs)
        init_sequences = np.array(self.initial_sequences[0])
        obs = np.concatenate(self.values)
        init_obs = np.concatenate(self.initial_values)
        artifact_uri = mlflow.active_run().info.artifact_uri
        seq_array_artifact_path = Path(artifact_uri) / "sequences_observations.npz"
        np.savez(seq_array_artifact_path, x=sequences, x0=init_sequences, y=obs, y0=init_obs)
        mlflow.log_artifact(seq_array_artifact_path)
        # mlflow.log_dict(self.additional_metrics, "additional_metrics.json")
        mlflow.end_run()


if __name__ == '__main__':
    # TODO: set specific observer for specific experiment in designated environment
    results_tracking_uri = Path(corel.__file__).parent.parent.parent.resolve() / "results" / "mlruns"
    set_observer(PoliBaseMlFlowObserver(tracking_uri=results_tracking_uri), 
                conda_environment_location="poli__lambo",
                observer_name="PoliBaseMlFlowObserver")