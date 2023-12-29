__author__ = 'Simon Bartels'

import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from gpflow import default_float
from poli.core.problem_setup_information import ProblemSetupInformation
from trieste.acquisition.optimizer import automatic_optimizer_selector, generate_continuous_optimizer
from trieste.space import SearchSpaceType, TaggedProductSearchSpace, Box
from tensorflow_probability.python.distributions import Categorical

from corel.util.k_best import KeepKBest
from corel.weightings.vae.cbas.cbas_factory import CBASVAEFactory


#from corel.weightings.vae.cbas_vae_wrapper import CBASVAEWrapper


class ContinuousLatentSpaceParameterizationOptimizerFactory:
    def __init__(self, problem_info: ProblemSetupInformation, batch_size=1, samples_from_proposal=50, 
                ss_lower_lim=-3., ss_upper_lim=3.):
        assert(problem_info.sequences_are_aligned())
        assert(samples_from_proposal >= batch_size)
        self.batch_size = batch_size
        self.samples_from_proposal = samples_from_proposal
        self.L = problem_info.get_max_sequence_length()
        self.AA = len(problem_info.get_alphabet())
        self.vae = CBASVAEFactory().create(problem_info)
        self.search_space_limits = (ss_lower_lim, ss_upper_lim)

    def create(self):
        def latent_optimizer(search_space: SearchSpaceType, acquisition_function, on_fail_retry_n: int=3) -> tf.Tensor:
            def make_ac():
                def ac(z):
                    assert len(z.shape) == 3 and z.shape[1] == 1
                    z = tf.reshape(z, [z.shape[0], z.shape[-1]])
                    return acquisition_function(tf.reshape(self.vae.decode(z), [z.shape[0], 1, self.L * self.AA]))
                return ac
            acquisition = make_ac()
            search_dim = self.vae.vae.latentDim_
            searchspace = Box(lower=self.search_space_limits[0] * tf.ones(search_dim), upper=self.search_space_limits[1] * tf.ones(search_dim))
            bestk = KeepKBest(self.batch_size, copy=lambda x: x.copy())
            for _ in range(on_fail_retry_n):
                try:
                    z = generate_continuous_optimizer(
                            num_initial_samples=1,
                            num_optimization_runs=1,
                            num_recovery_runs=0
                        )(searchspace, acquisition)
                    p = self.vae.decode(z)
                    bestk = self.get_best_of_k(self.samples_from_proposal, p, acquisition_function, bestk)
                except Exception as e: # TODO: this statement is waaaay too broad. FIXME
                    warnings.warn("An optimization attempt failed with exception " + str(e))
            selected_seqs, vals = bestk.get()
            selected_seqs_tensor = tf.constant(np.concatenate(selected_seqs.tolist(), dtype=np.int64)) # cast int64 explicitly
            return selected_seqs_tensor
        return latent_optimizer

    def get_best_of_k(self, k, P, acquisition_function, bestk: KeepKBest) -> KeepKBest:
        seq = tf.argmax(P, axis=-1)
        atom = self._seq_to_atom(seq)
        val = acquisition_function(atom)
        bestk.new_val(val, seq.numpy())
        dist = Categorical(P)
        for i in range(k):
            seq_ = dist.sample()
            val_ = acquisition_function(self._seq_to_atom(seq_))
            bestk.new_val(val_, seq_.numpy())
        return bestk

    def get_best_of_single_site_mutations(self, P, acquisition_function, bestk: KeepKBest) -> KeepKBest:
        seq = tf.argmax(P, axis=-1)
        atom = self._seq_to_atom(seq)
        val = acquisition_function(atom)
        bestk.new_val(val, seq.numpy())

        seq_ = seq.numpy().copy()
        for l in range(P.shape[1]):
            for a in range(1, P.shape[2]):
                seq_[0, l] = a
                val_ = acquisition_function(self._seq_to_atom(seq_))
                bestk.new_val(val_, seq_)
            seq_[0, l] = seq[0, l].numpy()
        return bestk

    def _seq_to_atom(self, x):
        return tf.reshape(tf.one_hot(x, depth=self.AA, axis=-1, dtype=default_float()), [x.shape[0], 1, self.L*self.AA])
