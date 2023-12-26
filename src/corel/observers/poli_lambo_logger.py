__author__ = 'Simon Bartels'

from corel.observers.logger import MLFlowLogger

"""
This Observer requires a lambo-specific environment with [lambo ; torch ; botorch] installed.
Use fx poli__lambo environment or install the environment specified in https://github.com/samuelstanton/lambo/tree/main/lambo
"""
from pathlib import Path
import numpy as np
from typing import Tuple
import torch
from botorch.utils.multi_objective import infer_reference_point, Hypervolume, is_non_dominated

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.registry import set_observer
from poli.core.util.abstract_observer import AbstractObserver

#from lambo.optimizers.pymoo import Normalizer
from corel.observers.lambo_imports.normalizer import Normalizer
import corel
from corel.observers import HD_PREV, HD_WT, HD_MIN, SEQUENCE, BLACKBOX, MIN_BLACKBOX, ABS_HYPER_VOLUME, REL_HYPER_VOLUME
#from corel.observers.logger import log, initialize_logger, finish, log_sequence


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
    norm_pareto_targets = hypercube_transform(y0[idx, ...])
    normed_ref_point = -infer_reference_point(-torch.tensor(norm_pareto_targets)).numpy()
    return normed_ref_point, transform
    

class PoliLamboLogger(AbstractObserver): 
    def __init__(self, tracking_uri: Path=None):
        self.info: ProblemSetupInformation = None
        self.tracking_uri = tracking_uri if tracking_uri is None else tracking_uri
        self.step = 1
        self.sequences = []
        self.vals = []
        self.wt = None
        self.initial_pareto_front_volume = None
        self.transformed_pareto_volume_ref_point = None
        self.transform = None

        # the following values are specific to the LamBO RFP problem
        #target_min = torch.tensor([-12008.24754087, -74.7978])
        #target_range = torch.tensor([3957.58754182, 156.8002])
        #self.lambo_transform = Normalizer(
        #    loc=target_min + 0.5 * target_range,
        #    scale=target_range / 2.,
        #)
        self.lambo_transform = None
        self.transformed_lambo_ref_point = None #torch.tensor([-0.11583198, 0.46189176])
        self.lambo_initial_pareto_front_volume = None  # 0.6751
        self.lambo_values = []
        self.logger = None

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        assert(y.shape[0] == 1)  # can process only one datapoint at a time
        if context is None or True:
            self.step += 1
            self.sequences.append(x)
            self.vals.append(y[0, :])
            self.lambo_values.append(y[0, :])
            self.logger.log({BLACKBOX + str(i): y[0, i] for i in range(y.shape[1])}, step=self.step)
            ymat = np.array(self.vals)
            mins = np.min(ymat, axis=0)
            assert(mins.shape[0] == y.shape[1])
            self.logger.log({MIN_BLACKBOX + str(i): mins[i] for i in range(y.shape[1])}, step=self.step)
            if self.info.sequences_are_aligned() and len(self.sequences) > 1:
                # TODO: monitor statistics for the unaligned case
                self.logger.log({HD_PREV: np.sum((self.sequences[-2] - x) != 0),
                     HD_WT: np.sum((self.wt - x) != 0),
                     HD_MIN: np.min(np.sum(np.vstack(self.sequences[:-1]) - x != 0, axis=-1)),
                     }, step=self.step, verbose=True)
            if self.transformed_pareto_volume_ref_point is not None and y.shape[1] > 1:
                new_volume = self._compute_hyper_volume(ymat)
                self.logger.log({ABS_HYPER_VOLUME: new_volume, REL_HYPER_VOLUME: new_volume / self.initial_pareto_front_volume},
                    step=self.step)
                self.logger.log({"LAMBO_REL_HYPER_VOLUME": self._compute_lambo_hyper_volume(np.array(self.lambo_values)) / self.lambo_initial_pareto_front_volume},
                    step=self.step)
            self.logger.log_sequence(x, step=self.step, verbose=True)

    def initialize_observer(self, problem_setup_info: ProblemSetupInformation, caller_info: dict, x0, y0, seed) -> object:
        self._initialize_logger()
        self.wt = x0[:1, ...]
        self.info = problem_setup_info
        run = self.logger.initialize_logger(problem_setup_info, caller_info, seed)
        self._add_initial_observations(x0, y0)
        # when not calling from here, returning the run would cause an exception!
        if "JSD_METHOD" in caller_info.keys():
            return run
        return None

    def _initialize_logger(self):
        self.logger = MLFlowLogger()

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

    def _compute_lambo_hyper_volume(self, all_y: np.ndarray) -> float:
        """
        Compute Hypervolume as a maximization problem.
        NOTE: This hypervolume is LaMBO specific, since reference point are LaMBO specific.
        Take transformed lambo reference points, compute w.r.t. normalized pareto targets.
        NOTE: Reference BoTorch implementation (LaMBO) used.
        NOTE: y observations are not negated for normalized pareto targets.
        """
        tymat = torch.Tensor(all_y)
        idx = is_non_dominated(-tymat).numpy()
        norm_pareto_targets = self.lambo_transform(all_y[idx, ...]).clone().detach()
        # this implementation of volume computation assumes maximization
        return Hypervolume(-self.transformed_lambo_ref_point).compute(-norm_pareto_targets)

    def _add_initial_observations(self, x0, y0):
        self.step = -len(x0)
        for i in range(y0.shape[0]):
            self.observe(x0[i:i+1, ...], y0[i:i + 1, ...])
        # the assertion assumes that the step increase is executed in the beginning of #observe
        assert(self.step == 0)

        if y0.shape[1] > 1:
            target_min = torch.min(y0, axis=0)
            target_range = torch.max(y0, axis=0) - target_min
            self.lambo_transform = Normalizer(
                loc=target_min + 0.5 * target_range,
                scale=target_range / 2.,
            )
            idx = is_non_dominated(-torch.Tensor(y0)).numpy()
            # RESET LamBO values
            self.lambo_values = y0[idx, :].numpy().tolist()
            transformed_pareto_volume_ref_point, self.transform = get_pareto_reference_point(y0)
            self.transformed_pareto_volume_ref_point = torch.Tensor(transformed_pareto_volume_ref_point)
            self.initial_pareto_front_volume = self._compute_hyper_volume(y0)
            self.logger.log({ABS_HYPER_VOLUME: self.initial_pareto_front_volume}, step=self.step)
            lambo_norm_pareto_targets = self.lambo_transform(y0[idx, ...])
            self.transformed_lambo_ref_point = -infer_reference_point(-torch.tensor(lambo_norm_pareto_targets)).numpy()
            lambo_initial_pareto_front_volume = self._compute_lambo_hyper_volume(np.array(self.lambo_values))
            self.logger.log({"LAMBO_ABS_HYPER_VOLUME": lambo_initial_pareto_front_volume}, step=self.step)
            self.lambo_initial_pareto_front_volume = lambo_initial_pareto_front_volume

    def finish(self) -> None:
        self.logger.finish()


if __name__ == '__main__':
    results_tracking_uri = Path(corel.__file__).parent.parent.parent.resolve() / "results" / "mlruns"
    set_observer(observer=PoliLamboLogger(), 
                conda_environment_location="poli__lambo",
                observer_name="PoliLamboLogger")
