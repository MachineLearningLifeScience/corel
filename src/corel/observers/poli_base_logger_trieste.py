from pathlib import Path
import numpy as np
import mlflow
import tensorflow as tf
import trieste
from trieste.acquisition.multi_objective.pareto import Pareto
from trieste.acquisition.multi_objective.dominance import non_dominated
from poli.core.util.abstract_observer import AbstractObserver
from poli.core.abstract_problem_factory import ProblemSetupInformation
from logging import log
from corel.observers import BLACKBOX, MIN_BLACKBOX, HD_PREV, HD_WT, HD_MIN, ABS_HYPER_VOLUME, REL_HYPER_VOLUME


STARTING_N = "n_D0"
ALGORITHM = "ALGORITHM"
BATCH_SIZE = "BATCH_SIZE"


class PoliBaseMlFlowObserver(AbstractObserver):
    def __init__(self, tracking_uri: Path) -> None:
        self.step = 0
        self.tracking_uri = tracking_uri
        self.values = []
        self.sequences = []
        self.info = None

        #TODO: do we require all of these metrics?
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
        if "run_id" in caller_info:
            run_id = caller_info["run_id"]
        else:
            run_id = None

        if "experiment_id" in caller_info:
            experiment_id = caller_info["experiment_id"]
        else:
            experiment_id = None
        
        self.info = problem_setup_info

        # Sets up the MLFlow experiment
        # Is there an experiment running at the moment?
        if mlflow.active_run() is not None:
            # If so, continue to log in it.
            mlflow.set_experiment(mlflow.active_run().info.experiment_name)
        else:
            # If not, create a new one.
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.start_run(run_id=run_id, experiment_id=experiment_id)

        mlflow.log_params(
            {
                "name": problem_setup_info.name,
                "max_sequence_length": problem_setup_info.max_sequence_length,
                "alphabet": problem_setup_info.alphabet,
            }
        )

        mlflow.log_param("x0", x0)
        mlflow.log_param("y0", y0)
        mlflow.log_param("seed", seed)
        mlflow.log_param(ALGORITHM, caller_info.get(ALGORITHM))
        mlflow.log_param(STARTING_N, caller_info.get(STARTING_N))
        mlflow.log_param(BATCH_SIZE, caller_info.get(BATCH_SIZE))

    def observe(self, x: np.ndarray, y: np.ndarray, context=None) -> None:
        mlflow.log_metric("y", y, step=self.step)
        self.sequences.append(x)
        self.values.append(y[0, :])

        if context is not None:
            for key, value in context.items():
                mlflow.log_metric(key, value, step=self.step)
        elif context is None or True:
            log({BLACKBOX + str(i): y[0, i] for i in range(y.shape[1])}, step=self.step)
            ymat = np.array(self.values)
            mins = np.min(ymat, axis=0)
            assert(mins.shape[0] == y.shape[1])
            log({MIN_BLACKBOX + str(i): mins[i] for i in range(y.shape[1])}, step=self.step)
            if self.transformed_pareto_volume_ref_point is not None and y.shape[1] > 1:
                new_volume = self._compute_hyper_volume(ymat)
                log({ABS_HYPER_VOLUME: new_volume, 
                    REL_HYPER_VOLUME: new_volume / self.initial_pareto_front_volume},
                    step=self.step)
        self.step += 1

    def _add_initial_observations(self, x0, y0):
        # TODO make these operations tf/trieste specific
        self.step = -len(x0)
        for i in range(y0.shape[0]):
            self.observe(x0[i:i+1, ...], y0[i:i + 1, ...])
        # the assertion assumes that the step increase is executed in the beginning of #observe
        assert(self.step == 0)
        if y0.shape[1] > 1:
            pareto = Pareto(tf.Tensor(y0))
            transformed_pareto_volume_ref_point = pareto.get_reference_point(y0)
            self.transformed_pareto_volume_ref_point = tf.constant(transformed_pareto_volume_ref_point)
            self.initial_pareto_front_volume = self._compute_hyper_volume(y0) # TODO: sign
            log({ABS_HYPER_VOLUME: self.initial_pareto_front_volume}, step=self.step)
        
    def _compute_hyper_volume(self, all_y: np.ndarray) -> float:
        # TODO make these operations tf/trieste specific
        # TODO test
        # TODO test exactness against botorch HV
        tymat = tf.Tensor(all_y)
        d_mask = non_dominated(-tymat).numpy()
        # the procedure assumes maximization
        # ref_point = -infer_reference_point(-tymat[idx, ...])
        # Not the same but a BoTorch implementation of the same algorithm as in LaMBO
        pareto = Pareto(tymat, d_mask) # TODO: sign and computation
        return pareto.hypervolume_indicator(self.transformed_pareto_volume_ref_point)
    

    def finish(self) -> None:
        if isinstance(self.tracking_uri, Path):
            with open(self.tracking_uri / "sequences.npy", "wb") as f:
                np.save(f, self.sequences)

            mlflow.log_artifact(self.tracking_uri / "sequences.npy")

        mlflow.end_run()