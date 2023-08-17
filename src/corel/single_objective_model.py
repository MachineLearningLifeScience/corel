__author__ = 'Simon Bartels'

import warnings

import numpy as np
import tensorflow as tf
from gpflow import default_float, set_trainable
from gpflow.logdensities import multivariate_normal
from gpflow.functions import Constant
from gpflow.optimizers import Scipy
from gpflow.models import GPR
from gpflow.kernels import Exponential
from gpflow.utilities import add_likelihood_noise_cov
from trieste.data import Dataset
from trieste.models import TrainableProbabilisticModel
from trieste.models.gpflow import GaussianProcessRegression
from trieste.types import TensorType


class ProteinModel(TrainableProbabilisticModel):
    """
    Model over the product simplex of size length x AA.
    """
    def __init__(self, distribution, AA):
        self._optimized = False
        self.distribution = distribution
        self.AA = AA
        self.ps = None
        self.L = None  # Cholesky of the kernel matrix
        self.alpha = None
        # TODO: proper initialization of lengthscale!
        self.log_length_scale = tf.Variable(tf.ones(1, dtype=default_float()))
        self.amplitude = None
        self.kernel_mean = None
        self.model = None
        self.dataset = None
        #self.noise = 1e-8
        self.noise = tf.Variable(1e-8 * tf.ones(1, dtype=default_float()))

    def _predict(self, query_points: TensorType) -> TensorType:
        assert(self._optimized)
        if query_points.dtype.is_integer:
            # TODO: this should maybe be an input transformation?
            query_points = tf.one_hot(query_points, self.AA, dtype=default_float())
            # flatten
            query_points = tf.reshape(query_points, [query_points.shape[0], query_points.shape[1] * query_points.shape[2]])
        t = np.empty([query_points.shape[0], self.L.shape[0]])
        for i in range(query_points.shape[0]):
            t[i, :] = self._predict_single_query(query_points[i:i+1, ...])
        return tf.constant(t)

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
        # I have removed the amplitude here. For the mean it cancels and we apply it later for the covariance
        cov = tf.exp(-tf.sqrt(squared_hellinger_distance) / tf.exp(self.log_length_scale))
        temp = tf.transpose(tf.linalg.triangular_solve(self.L, tf.transpose(cov), lower=True))
        return temp

    def predict(self, query_points: TensorType) -> tuple[TensorType, TensorType]:
        temp = self._predict(query_points)
        mean = temp @ self.alpha + self.kernel_mean
        # TODO: no noise, correct?
        variance = self.amplitude * (tf.ones([query_points.shape[0], 1], dtype=default_float()) - tf.expand_dims(tf.reduce_sum(tf.square(temp), axis=-1), axis=1))
        return mean, variance

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        raise NotImplementedError("not implemented")
        temp = self._predict(query_points)
        mean = temp @ self.alpha + self.kernel_mean
        # TODO: noise?
        variance = self.amplitude * (prior_cov - temp @ tf.transpose(temp))
        samples = mean + tf.linalg.cholesky(variance) @ tf.random.normal([query_points.shape[0, num_samples]])
        return samples

    def update(self, dataset: Dataset) -> None:
        self._optimized = False

    def optimize(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self._optimized = True
        # transform query points to one_hot? No, better do that in the distribution. HMMs may prefer that
        # TODO: the step below is a huge bottleneck! It's repeated for all models but pretty expensive!
        self.ps = self.distribution(dataset.query_points)
        # TODO: try median?
        self.log_length_scale.assign(tf.math.log(tf.reduce_mean(self.ps) * tf.ones(1, dtype=default_float())))
        squared_hellinger_distance = _hellinger_distance(self.ps)
        print("squared Hellinger distance: \n" + str(squared_hellinger_distance.numpy()))
        optimizer = Scipy()

        def make_closure():
            def opt_criterion():
                ks = _k(squared_hellinger_distance, self.log_length_scale, self.noise)
                try:
                    L = tf.linalg.cholesky(ks)
                except Exception as e:
                    raise e
                m, r = get_mean_and_amplitude(L, dataset.observations)
                log_prob = multivariate_normal(dataset.observations, m * tf.ones([L.shape[0], 1], default_float()),
                                               tf.sqrt(r) * L)
                return -tf.reduce_sum(log_prob)

            return opt_criterion

        optimizer.minimize(
            make_closure(),
            [self.log_length_scale, self.noise],
            # options=dict(maxiter=reduce_in_tests(1000)),
        )
        ks = _k(squared_hellinger_distance, self.log_length_scale, self.noise)
        self.L = tf.linalg.cholesky(ks)
        # L_ = tf.linalg.cholesky(K)
        self.kernel_mean, self.amplitude = get_mean_and_amplitude(self.L, dataset.observations)
        # self.length_scale = model.kernel.lengthscales * tf.sqrt(2 * tf.ones(1, dtype=self.ps.dtype))  # is one-dimensional because the model is
        # self.length_scale = model.kernel.lengthscales
        print("covariance matrix:\n" + str(ks.numpy()))
        print("length scale: " + str(np.exp(self.log_length_scale.numpy())))
        self.alpha = tf.linalg.triangular_solve(self.L, dataset.observations - self.kernel_mean, lower=True)
        # self.model = model


def _hellinger_distance(ps):
    """
    This function assumes that elements with exactly the same probability are the same!
    :param ps:
    :type ps:
    :return:
    :rtype:
    """
    squared_hellinger_distance = (ps + tf.transpose(ps)) / 2
    # if we have the same point twice, set the distance to 0 there
    squared_hellinger_distance = tf.where(
        ps - tf.transpose(ps) == 0,
        tf.zeros_like(squared_hellinger_distance), squared_hellinger_distance)
    return squared_hellinger_distance


def _k(HD, log_lengthscale, noise):
    #K = (X + tf.transpose(X)) / 2
    #K = tf.linalg.set_diag(K, tf.zeros(K.shape[0], dtype=K.dtype))
    K = tf.math.exp(-tf.sqrt(HD) / tf.exp(log_lengthscale))
    K = K + noise * tf.eye(K.shape[0], dtype=K.dtype)
    return K


def get_mean_and_amplitude(L, Y):
    ones = tf.linalg.triangular_solve(L, tf.ones_like(Y))
    alpha = tf.linalg.triangular_solve(L, Y)
    n = tf.reduce_sum(tf.square(ones))
    m = tf.reduce_sum(ones * alpha) / n
    #r = tf.reduce_sum(tf.square(alpha - m * tf.ones_like(Y)))
    # Above amplitude estimator seems to be missing a normalization!
    # below is the factor as set in Jones et al. 1998
    r = tf.reduce_sum(tf.square(alpha - m * tf.ones_like(Y))) / Y.shape[0]
    return m, r
