__author__ = "RM"

import logging
from copy import deepcopy
from inspect import Parameter
from logging import info
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from gpflow import Parameter, default_float
from gpflow.kernels import Product
from gpflow.models import GPR
from gpflow.optimizers import Scipy
from gpflow.utilities import positive, to_default_float
from trieste.data import Dataset
from trieste.models import TrainableProbabilisticModel
from trieste.types import TensorType

from corel.kernel.hellinger import get_mean_and_amplitude
from corel.kernel.weighted_hellinger import WeightedHellinger


class LVMModel(TrainableProbabilisticModel):
    """
    Model over the product of weighted HKs
    NOTE: this model is currently implemented for the 1D case!
    """
    def __init__(self, distribution: tf.Tensor, AA: int, L: int, unlabelled_data: Optional[TensorType]=None, batch_size: int=1, nan_penalty: float=10.) -> None:
        """
        Instantiate LVMModel class.
            Distribution is callable object embedding distribution
            unlabelled_data is data used to train the distribution object
        """
        super().__init__()
        self.distribution = distribution
        self.encoder = self.distribution.vae.vae.encoder_
        self.decoder = self.distribution.vae.vae.decoder_
        self.aa = AA
        self.len = L
        self.batch = batch_size
        self.ps = None  # distributions of X
        self.X = None  # valid sequences from dataset
        self.y = None  # valid observations from data
        self.L = None  # decomposed matrix
        self.alphas = None
        self.kern_mean = None
        self.kern_amplitude = None
        self.variance = None
        self.log_lengthscale = None
        self.noise = None
        self._optimized = False
        self.model = None
        self._kernel = None
        self.reference_data = self._one_hot_encode_ints(unlabelled_data)
        self.nan_penalty = nan_penalty

    def _one_hot_encode_ints(self, data) -> TensorType:
        """
        one-hot encode sequence of integers
        Return encoded sequence vector
        """
        one_hot_pts = tf.one_hot(data, self.aa, dtype=default_float())
        if one_hot_pts.shape[-1] != self.aa and len(one_hot_pts.shape) == 2:
            one_hot_pts = tf.reshape(one_hot_pts, [one_hot_pts.shape[0], one_hot_pts.shape[1] // self.aa, self.aa])
        return one_hot_pts

    def _compute_weight_matrix_from_reference_sequences(self) -> TensorType:
        """
        Encode and obtain decoding distributions for reference_data
            ie. unlabelled data used to train underlying LVM
        """
        if self.reference_data is not None:
            logging.info(f"Computing weighting for {len(self.reference_data)} reference sequences")
            encoding_mu = self.encoder(self.reference_data)[0]
            weighting_matrix = self.decoder(encoding_mu)
        else:
            raise RuntimeError("No weighting matrix provided!\nSpecify unlabelled_data as input.")
        return weighting_matrix


    def _init_model(self, init_data_var=0.01) -> GPR:
        kernels = []
        weighting_matrix = self._compute_weight_matrix_from_reference_sequences()
        init_len = max(np.sqrt(np.median(weighting_matrix)), np.exp(-350))
        lengthscale = Parameter(init_len, transform=positive(), name="lengthscale")
        for _p in weighting_matrix:
            assert _p.shape[0] == self.len and _p.shape[1] == self.aa
            kernels.append(WeightedHellinger(w=_p, AA=self.aa, L=self.len, lengthscale=lengthscale))
        info("Composing wHK to product kernel")
        # TODO: enforce shared length-scale
        self._kernel = Product(kernels=kernels, name="product_whk")
        model = GPR((self.ps, self.y), kernel=self._kernel)
        model.likelihood.variance = Parameter(init_data_var, transform=positive(), name="noise") # TODO: better initial value here
        return model

    def sample(self, query_points: TensorType, num_samples: int) -> TensorType:
        # TODO: sample posterior predictive of the model
        raise NotImplementedError("Not implemented")

    def log(self, dataset: Optional[Dataset]=None) -> None:
        pass

    def update(self, dataset: Dataset, batch_size=1) -> None:
        if self.nan_penalty:
            valid_observations = deepcopy(dataset.observations.numpy())
            nan_idx = np.isnan(dataset.observations.numpy())[...,0]
            info(f"Replacing {nan_idx.sum()} NaN values with penalty fixed value")
            valid_observations[nan_idx] = self.nan_penalty  # assign NaN value a positive value
            valid_query_points = dataset.query_points  # all query points are valid now
        else: # take out NaN values which break GP fit/opt
            valid_idx = np.isfinite(dataset.observations.numpy())[..., 0]
            valid_query_points = dataset.query_points[valid_idx, ...]
            valid_observations = dataset.observations[valid_idx, ...]  # NOTE: not all query points will be valid!
        if self.X is not None:
            oldN = self.X.shape[0]
            _valid = self._test_validity_data_observations(query_points=valid_query_points[:oldN, ...], observations=valid_observations[:oldN, ...])
        else:
            oldN = 0
        info("Computing sequence probabilities...")
        one_hot_querypoints = tf.one_hot(valid_query_points[oldN:, ...], self.aa)
        psnew = to_default_float(self.decoder(self.encoder(one_hot_querypoints)[0]))
        psnew = tf.reshape(psnew, shape=(psnew.shape[0], self.len*self.aa))
        if self.ps is None:
            self.ps = psnew
        else:
            self.ps = tf.concat([self.ps, psnew], axis=0)
        assert self.ps.shape[0] == valid_observations.shape[0] and int(self.ps.shape[1]) == int(self.len*self.aa)
        self.X = valid_query_points
        self.y = valid_observations
        if self.model is None:
            self.model = self._init_model()
        self._optimized = False

    def _predict_on_query(self, ps: TensorType) -> Tuple[TensorType, TensorType]:
        """
        Input: probability (decoder) matrix
        Solve for posterior predictive of GP using cholesky decomposed data matrix ()
        return predictive mean and variance
        """
        # assert ps.shape[0] == 1
        ps = tf.reshape(ps, shape=(ps.shape[0], self.len*self.aa))
        # FIXME: predict_f fails due to batch shape mismatches within the posterior computation! Input [B, N, D] only [N, D] gets passed through, returns [N, N] shape-check fail!
        # pred_mean, pred_var = self.model.predict_f(ps[tf.newaxis,...]) # expects [B, N, D] inputs
        # manually compute posterior predictive
        X, Y = self.model.data
        prior_mean, prior_amp = get_mean_and_amplitude(L=self.L, Y=Y)
        alphas = tf.linalg.triangular_solve(self.L, Y-prior_mean, lower=True)
        K_Xx = self.model.kernel(X[tf.newaxis,...], ps[tf.newaxis,...]).reshape(X.shape[0], ps.shape[0]) # drop batch dimensions at index [0, 2]
        L_x_solve = tf.transpose(tf.linalg.triangular_solve(self.L, K_Xx, lower=True))
        # TODO: test predictive posterior computation
        pred_mean = L_x_solve @ alphas + prior_mean
        pred_var = prior_amp * tf.ones((ps.shape[0], 1), dtype=tf.float64) - tf.expand_dims(tf.reduce_sum(tf.square(L_x_solve), axis=-1), axis=1)
        return pred_mean, pred_var

    def predict(self, query_points: TensorType) -> Tuple[TensorType, TensorType]:
        """
        Obtain one-hot embedding for integer query, invoke prediction on query.
        Return predictive posterior mean and variance.
        """
        assert self._optimized , "Model not optimized!"
        info("Predict on query")
        if self.y.shape[1] > 1:
            raise NotImplementedError("Only 1D case implemented!\n Multiple output dimensions were provided.")
        if query_points.dtype.is_integer:
            query_points = tf.one_hot(query_points, self.aa, dtype=default_float()) # one-hot query encoding
            query_points = tf.reshape(query_points, shape=(query_points.shape[0], self.len, self.aa))
            query_points = to_default_float(self.decoder(self.encoder(query_points)[0])) # LVM encoding step
        assert query_points.dtype == default_float()
        pred_mean, pred_var = self._predict_on_query(query_points)
        pred_mean = pred_mean.reshape((query_points.shape[0], 1))
        pred_var = pred_var.reshape((query_points.shape[0], 1))
        return pred_mean, pred_var
    
    def _test_validity_data_observations(self, query_points: TensorType, observations: TensorType) -> bool:
        if isinstance(query_points, tf.Tensor):
            query_points = query_points.numpy()
        if isinstance(observations, tf.Tensor):
            observations = observations.numpy()
        valid_data = np.all(np.array(self.X) == query_points)
        valid_obs = np.all(np.array(self.y) == observations) # TODO: account for NaN values that break assert
        if not valid_data or not valid_obs:
            raise ValueError(f"Model points, observations invalid w.r.t. input data!\nX={self.X.shape} y={self.y.shape} against query={query_points.shape} obs={observations.shape}") # TODO: make specific
        return True

    def optimize(self, dataset: Dataset) -> None:
        if dataset.observations.shape[1] > 1:
            raise NotImplementedError("Only 1D case implemented!")
        # NOTE: below becomes a redundant check with replaced NaNs we ask for self.X==self.X
        # _valid_data = self._test_validity_data_observations(query_points=self.X, observations=self.y)
        opt = Scipy()
        opt.minimize(self.model.training_loss, 
                    self.model.trainable_variables, 
                    bounds=[(np.exp(-350), None), # constraint length_scale, prohibit values close to or equal zero
                        (0, 1)], # constrain noise to zero-one range
                    )
        X, Y = self.model.data
        self.L = tf.linalg.cholesky(self.model.kernel(X))
        _len, _var = self.model.trainable_parameters
        prior_mean, prior_amp = get_mean_and_amplitude(self.L, Y)
        self.noise = _var
        self.log_lengthscale = tf.math.log(_len)
        self.kern_mean = prior_mean
        self.kern_amplitude = prior_amp
        self.alpha = tf.linalg.triangular_solve(self.L, Y - prior_mean, lower=True)
        self._optimized = True

    def get_context(self) -> dict:
        """
        Context for hyperparameter logging
        """
        context = dict(
            alphas=[self.alpha.numpy()],
            len_scales=[np.exp(self.log_lengthscale.numpy())],
            amplitude=[self.kern_amplitude.numpy()],
            kernel_mu=[self.kern_mean.numpy()],
            noises=[self.noise.numpy()],
            nan_penalty=[self.nan_penalty],
        )
        return context