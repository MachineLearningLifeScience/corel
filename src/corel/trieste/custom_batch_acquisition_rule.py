__author__ = 'Simon Bartels'

from trieste.acquisition.rule import EfficientGlobalOptimization


class CustomBatchEfficientGlobalOptimization(EfficientGlobalOptimization):
    def __init__(self, builder, optimizer, num_query_points=1, initial_acquisition_function=None):
        # just to avoid the wrapping of the builder
        super().__init__(builder, optimizer, num_query_points=1, initial_acquisition_function=initial_acquisition_function)
        self._num_query_points = num_query_points
