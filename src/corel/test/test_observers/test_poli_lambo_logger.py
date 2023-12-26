import unittest

import numpy as np
from poli import objective_factory

from corel.observers.poli_lambo_logger import PoliLamboLogger
from corel.util.test_util.DummyLogger import DummyLogger


class PoliLamboLoggerTestCase(unittest.TestCase):
    def test_equivalence_when_seed_is0(self):
        problem = 'foldx_rfp_lambo'
        seed = 0
        problem_info, _, x0, y0, run_info = objective_factory.create(
            name=problem,
            seed=seed,
            caller_info=None,
            observer=None, # NOTE: This has to be the lambo specific logger
            force_register=False,
            parallelize=False, # NOTE: current setup DO NOT allow parallelization
        )
        observer = TestPoliLamboLogger()
        observer.initialize_observer(problem_info, dict(), x0, y0, seed)
        lambo_values = np.array([
             [-10376.84011515, -71.4708],
             [-10820.91136186, -55.6143],
             [-11558.62762577,  29.6978],
             [-11189.00587946, -39.8155],
             [-11445.82982225, -27.9617],
             [-10591.87684184, -61.8757]
        ])
        lambo_values.sort(axis=0)
        print(lambo_values)
        lambo_values_ = np.array(observer.lambo_values)
        lambo_values_.sort(axis=0)
        print(lambo_values_)
        np.testing.assert_array_almost_equal(lambo_values_, lambo_values)
        np.testing.assert_array_almost_equal(observer.transformed_lambo_ref_point.numpy(), np.array([-0.11583198, 0.46189176]))
        self.assertAlmostEqual(0.675107, observer.lambo_initial_pareto_front_volume)  # add assertion here


class TestPoliLamboLogger(PoliLamboLogger):
    def _initialize_logger(self):
        self.logger = DummyLogger()


if __name__ == '__main__':
    unittest.main()
