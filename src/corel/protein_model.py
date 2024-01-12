__author__ = 'Simon Bartels'

import logging
import warnings
from typing import Tuple, Optional
import numpy as np
import tensorflow as tf
from gpflow import default_float, set_trainable
from gpflow.logdensities import multivariate_normal
from gpflow.optimizers import Scipy
from trieste.data import Dataset
from trieste.models import TrainableProbabilisticModel
from trieste.types import TensorType
from typing import List

from corel.kernel.hellinger import get_mean_and_amplitude
from corel.kernel.hellinger import _hellinger_distance
from corel.kernel.hellinger import _k


class ProteinModel(TrainableProbabilisticModel):
    """
    Model over the product simplex of size length x AA.
    """
    def __init__(self, distribution, AA):
        self._optimized = False
        self.distribution = distribution
        self.AA = AA
        self.ps = None
        self.Ls = None  # Cholesky of the kernel matrix
        self.alphas = None
        self.log_length_scales = None
        self.amplitudes = None
        self.kernel_means = None
        self.dataset = None
        self.log_noises = None

    def log(self, dataset: Optional[Dataset] = None) -> None:
        pass

    def _predict(self, query_points: TensorType) -> List[List[TensorType]]:
        assert(self._optimized)
        if query_points.dtype.is_integer:
            query_points = tf.one_hot(query_points, self.AA, dtype=default_float())
            # flatten
            query_points = tf.reshape(query_points, [query_points.shape[0], query_points.shape[1] * query_points.shape[2]])
        temp = [self._predict_single_query(query_points[n:n+1, ...]) for n in range(query_points.shape[0])]
        num_tasks = self.dataset.observations.shape[1]
        return [tf.concat([temp[n][i] for n in range(query_points.shape[0])], axis=0) for i in range(num_tasks)]

    def _predict_single_query(self, query_points):
        assert(query_points.shape[0] == 1)
        # this implementation assumes that each query point is a distribution
        p_query = tf.reshape(query_points, [query_points.shape[0], query_points.shape[1] // self.AA, self.AA])
        ps = self.distribution(p_query)
        qs = np.prod(p_query.numpy()[0, np.arange(p_query.shape[1]), self.dataset.query_points.numpy()], axis=-1)
        qs = tf.constant(qs.reshape([query_points.shape[0], self.dataset.query_points.shape[0]]))
        #qs = tf.reduce_prod(tf.gather(query_points[0, ...], self.dataset.query_points, dim=-1))
        # BEWARE: this is NOT correct for atoms!
        # the equation below can become numerically negative
        #squared_hellinger_distance = ps / 2 - tf.transpose(self.ps) * tf.square(1 - tf.sqrt(qs)) / 2
        # the following equation seems to be better suitable
        squared_hellinger_distance = ps / 2 + tf.transpose(self.ps) * (0.5 - tf.sqrt(qs))
        assert(query_points.shape[0] == 1)
        if np.all(tf.square(query_points).numpy() == query_points.numpy()):
            #assert(qs.numpy() == 0.)  # only for not observed points
            # we are dealing with atoms
            squared_hellinger_distance = (ps + tf.transpose(self.ps)) / 2
            # if we have the same point twice, set the distance to 0 there
            squared_hellinger_distance = tf.where(
                tf.reduce_all(tf.cast(tf.argmax(p_query, axis=-1), self.dataset.query_points.dtype) - self.dataset.query_points == 0, axis=-1),
                tf.zeros_like(squared_hellinger_distance), squared_hellinger_distance)
        num_tasks = self.dataset.observations.shape[1]
        # I have removed the amplitude here. For the mean it cancels and we apply it later for the covariance
        cov = [tf.exp(-tf.sqrt(squared_hellinger_distance) / tf.exp(self.log_length_scales[i])) for i in range(num_tasks)]
        temp = [tf.transpose(tf.linalg.triangular_solve(self.Ls[i], tf.transpose(cov[i]), lower=True)) for i in range(num_tasks)]
        return temp

    def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        temp = self._predict(query_points)
        num_tasks = self.dataset.observations.shape[1]
        mean = tf.concat([temp[i] @ self.alphas[i] + self.kernel_means[i] for i in range(num_tasks)], axis=-1)
        # TODO: no noise, correct?
        variance = tf.concat([self.amplitudes[i] * (tf.ones([query_points.shape[0], 1], dtype=default_float()) - tf.expand_dims(tf.reduce_sum(tf.square(temp[i]), axis=-1), axis=1)) for i in range(num_tasks)], axis=-1)
        return mean, variance

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        raise NotImplementedError("not implemented")

    def update(self, dataset: Dataset) -> None:
        if self.dataset is not None:
            oldN = self.dataset.query_points.shape[0]
            assert(np.all(self.dataset.query_points.numpy() == dataset.query_points.numpy()[:oldN, ...]))
            assert(np.all(self.dataset.observations.numpy() == dataset.observations.numpy()[:oldN, ...]))
        else:
            oldN = 0
        print("getting probabilities of sequences")
        psnew = self.distribution(dataset.query_points[oldN:, ...])
        if self.ps is None:
            self.ps = psnew
        else:
            self.ps = tf.concat([self.ps, psnew], axis=0)
        self.dataset = dataset
        self._optimized = False

    def optimize(self, dataset: Dataset) -> None:
        assert(np.all(self.dataset.query_points.numpy() == dataset.query_points.numpy()))
        assert(np.all(self.dataset.observations.numpy() == dataset.observations.numpy()))
        # transform query points to one_hot? No, better do that in the distribution. HMMs may prefer that
        num_tasks = dataset.observations.shape[1]
        # NOTE: minimal initial value cannot be zero, but smallest possible computed value from p-vec
        initial_length_scale = max(np.sqrt(np.median(self.ps.numpy())), np.exp(-350)) # add small epsilon for numerical stability in case of close to zero ps
        self.log_length_scales = [tf.Variable(tf.math.log(initial_length_scale * tf.ones(1, dtype=default_float()))) for _ in range(num_tasks)]
        self.log_noises = [tf.Variable(tf.math.log(1e-3 * tf.ones(1, dtype=default_float()))) for _ in range(num_tasks)]
        squared_hellinger_distance = _hellinger_distance(self.ps)
        print("squared Hellinger distance: \n" + str(squared_hellinger_distance.numpy()))

        for i in range(num_tasks):
            optimizer = Scipy()

            def make_closure():
                def opt_criterion():
                    ks = _k(squared_hellinger_distance, self.log_length_scales[i], self.log_noises[i])
                    try:
                        L = tf.linalg.cholesky(ks)
                    except Exception as e:
                        logging.exception(e)
                        raise e
                    m, r = get_mean_and_amplitude(L, dataset.observations[:, i:i+1])
                    log_prob = multivariate_normal(dataset.observations[:, i:i+1], m * tf.ones([L.shape[0], 1], default_float()),
                                                   tf.sqrt(r) * L)
                    nll = -tf.reduce_sum(log_prob)
                    if not tf.math.is_finite(nll):
                        return tf.convert_to_tensor(np.inf)
                    return nll
                return opt_criterion

            optimizer.minimize(
                make_closure(),
                [self.log_length_scales[i], self.log_noises[i]],
                bounds = [(-350, None), # constraint log_length_scale, prohibit values close to or equal zero
                        (None, None)] # constraint log_noise
            )
            print(tf.math.exp(self.log_length_scales[i]))
            print(tf.math.exp(self.log_noises[i]))
        self.Ls = [tf.linalg.cholesky(_k(squared_hellinger_distance, self.log_length_scales[i], self.log_noises[i])) for i in range(num_tasks)]
        self.kernel_means = dict()
        self.amplitudes = dict()
        for i in range(num_tasks):
            self.kernel_means[i], self.amplitudes[i] = get_mean_and_amplitude(self.Ls[i], dataset.observations[:, i:i+1])
        print("length scales: " + str([np.exp(self.log_length_scales[i].numpy()) for i in range(num_tasks)]))
        print("noises: " + str([np.exp(self.log_noises[i].numpy()) for i in range(num_tasks)]))
        self.alphas = [tf.linalg.triangular_solve(self.Ls[i], dataset.observations[:, i:i+1] - self.kernel_means[i], lower=True) for i in range(num_tasks)]
        self._optimized = True

    def get_context(self) -> dict:
        """
        Provides context for logging hyperparameters.
        """
        num_tasks = self.dataset.observations.shape[1]
        context = dict(
            alphas=[_a.numpy() for _a in self.alphas],
            len_scales=[np.exp(self.log_length_scales[i].numpy()) for i in range(num_tasks)],
            amplitudes=[_a.numpy() for _a in self.amplitudes.values()],
            kernel_mu=[_ker_m.numpy() for _ker_m in self.kernel_means.values()],
            noises=[np.exp(self.log_noises[i].numpy()) for i in range(num_tasks)],
        )
        return context

