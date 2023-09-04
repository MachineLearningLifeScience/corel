__author__ = 'Simon Bartels'

import numpy as np

import torch

from poli.core.problem_setup_information import ProblemSetupInformation
from poli.core.util.abstract_observer import AbstractObserver

from external.lambo.Normalizer import Normalizer
from external.lambo.pareto_reference_point import get_pareto_reference_point
from util.constants import HD_PREV, HD_WT, HD_MIN, SEQUENCE, BLACKBOX, MIN_BLACKBOX, ABS_HYPER_VOLUME, REL_HYPER_VOLUME
from util.logger import log, initialize_logger, finish, log_sequence


"""
TODO: this class requires torch and lambo dependencies! FIXME
TODO: run in a separate lambo environment
"""

class PoliLogger(AbstractObserver):
    def __init__(self):
        self.step = 1
        self.sequences = []
        self.vals = []
        self.wt = None
        self.initial_pareto_front_volume = None
        self.transformed_pareto_volume_ref_point = None
        self.transform = None
        target_min = torch.tensor([-12008.24754087, -74.7978])
        target_range = torch.tensor([3957.58754182, 156.8002])
        self.lambo_transform = Normalizer(
            loc=target_min + 0.5 * target_range,
            scale=target_range / 2.,
        )
        self.transformed_lambo_ref_point = torch.tensor([-0.11583198, 0.46189176])
        self.lambo_initial_pareto_front_volume = 0.6751
        self.info: ProblemSetupInformation = None

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        assert(y.shape[0] == 1)  # can process only one datapoint at a time
        if context is None or True:
            self.step += 1
            self.sequences.append(x)
            self.vals.append(y[0, :])
            log({BLACKBOX + str(i): y[0, i] for i in range(y.shape[1])}, step=self.step)
            ymat = np.array(self.vals)
            mins = np.min(ymat, axis=0)
            assert(mins.shape[0] == y.shape[1])
            log({MIN_BLACKBOX + str(i): mins[i] for i in range(y.shape[1])}, step=self.step)
            if self.info.sequences_are_aligned() and len(self.sequences) > 1:
                # TODO: monitor statistics for the unaligned case
                log({#SEQUENCE: x,
                     HD_PREV: np.sum((self.sequences[-2] - x) != 0),
                     HD_WT: np.sum((self.wt - x) != 0),
                     HD_MIN: np.min(np.sum(np.vstack(self.sequences[:-1]) - x != 0, axis=-1)),
                     }, step=self.step, verbose=True)
            if self.transformed_pareto_volume_ref_point is not None and y.shape[1] == 2:
                new_volume = self._compute_hyper_volume(ymat)
                log({ABS_HYPER_VOLUME: new_volume, REL_HYPER_VOLUME: new_volume / self.initial_pareto_front_volume},
                    step=self.step)
                log({"LAMBO_REL_HYPER_VOLUME": self._compute_lambo_hyper_volume(ymat) / self.lambo_initial_pareto_front_volume},
                    step=self.step)

            log_sequence(x, step=self.step, verbose=True)

    def initialize_observer(self, problem_setup_info: ProblemSetupInformation, caller_info: dict, x0, y0, seed) -> object:
        #assert(isinstance(x0, list))
        self.wt = x0[:1, ...]
        self.info = problem_setup_info
        #for n in range(len(x0)):
        #    self.sequences.append(x0[n])
        run = initialize_logger(problem_setup_info, caller_info, seed)
        self._add_initial_observations(x0, y0)
        # when not calling from here, returning the run would cause an exception!
        if "JSD_METHOD" in caller_info.keys():
            return run
        return None

    def _compute_hyper_volume(self, all_y: np.ndarray) -> float:
        tymat = torch.Tensor(all_y)
        idx = is_non_dominated(-tymat).numpy()
        # the procedure assumes maximization
        # ref_point = -infer_reference_point(-tymat[idx, ...])
        # Not the same but a BoTorch implementation of the same algorithm as in LaMBO
        # No negation all_y!
        norm_pareto_targets = self.transform(all_y[idx, ...])
        #print(norm_pareto_targets)
        # this implementation of volume computation assumes maximization
        return Hypervolume(-self.transformed_pareto_volume_ref_point).compute(torch.tensor(-norm_pareto_targets))

    def _compute_lambo_hyper_volume(self, all_y: np.ndarray) -> float:
        tymat = torch.Tensor(all_y)
        idx = is_non_dominated(-tymat).numpy()
        # the procedure assumes maximization
        # ref_point = -infer_reference_point(-tymat[idx, ...])
        # Not the same but a BoTorch implementation of the same algorithm as in LaMBO
        # No negation all_y!
        norm_pareto_targets = self.lambo_transform(all_y[idx, ...])
        #print(norm_pareto_targets)
        # this implementation of volume computation assumes maximization
        return Hypervolume(-self.transformed_lambo_ref_point).compute(torch.tensor(-norm_pareto_targets))

    def _add_initial_observations(self, x0, y0):
        self.step = -len(x0)
        for i in range(y0.shape[0]):
            self.observe(x0[i:i+1, ...], y0[i:i + 1, ...])
        # the assertion assumes that the step increase is executed in the beginning of #observe
        assert(self.step == 0)
        if y0.shape[1] == 2:
            #transformed_pareto_volume_ref_point, self.transform = get_pareto_reference_point(y0)
            # TODO: REMOVE!
            transformed_pareto_volume_ref_point, self.transform = get_pareto_reference_point(y0)
            self.transformed_pareto_volume_ref_point = torch.Tensor(transformed_pareto_volume_ref_point)
            self.initial_pareto_front_volume = self._compute_hyper_volume(y0)
            log({ABS_HYPER_VOLUME: self.initial_pareto_front_volume}, step=self.step)

    def finish(self) -> None:
        finish()