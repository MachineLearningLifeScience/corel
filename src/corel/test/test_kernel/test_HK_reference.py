from typing import Callable
import pytest
import inspect
import numpy as np
from corel.kernel.hellinger import get_mean_and_amplitude
from corel.kernel.hellinger import _hellinger_distance
from corel.kernel.hellinger import _k
from corel.kernel import HellingerReference
from corel.kernel import Hellinger
from corel.kernel import WeightedHellinger
import tensorflow as tf
import matplotlib.pyplot as plt

# define test sequences and test alphabet and test weighting distributions
SEED=12
N = 20
L = 15
AA = 3
np.random.seed(SEED)

simulated_decoding_distributions = np.stack([
        np.random.dirichlet(np.ones(AA), L) for _ in range(N)
     ])

simulated_weighting_vec = np.random.dirichlet(np.ones(AA), L)


@pytest.mark.parametrize("dist", [simulated_decoding_distributions])#, simulated_weighting_vec])
def test_simulated_dist_is_probabilities(dist):
    summed_dist = np.sum(dist, axis=-1)
    np.testing.assert_almost_equal(summed_dist, np.ones((N, L)))
    np.testing.assert_array_less(dist, np.ones_like(dist))
    np.testing.assert_array_less(np.zeros_like(dist), dist)


def test_simulated_w_vec_is_probabilities():
    summed_dist = np.sum(simulated_weighting_vec, axis=-1)
    np.testing.assert_almost_equal(summed_dist, np.ones_like(summed_dist))


def really_naive_r(p_x: np.ndarray, q_y: np.ndarray):
    assert p_x.shape[0] == q_y.shape[0] and p_x.shape[1] == q_y.shape[1], "Input distributions inconsistent"
    L = p_x.shape[0]
    AA = p_x.shape[1]
    dist_prod_sum_across_sequence = 0.
    for l in range(L):
        for a in range(AA):
            dist_prod_sum_across_sequence += np.sqrt(p_x[l, a] * q_y[l, a])
    return np.sqrt(1 - dist_prod_sum_across_sequence)


def naive_r(p_x: np.ndarray, q_y: np.ndarray):
    """
    p, q are probability distributions (ie. decoder distributions)
    """
    # assumption sequences x , y are of shape (L, |AA|) with L seq-length and |AA| size of alphabet
    assert p_x.shape[0] == q_y.shape[0] and p_x.shape[1] == q_y.shape[1], "Input distributions inconsistent"
    if np.all(p_x == q_y):
        # the Hellinger distance between equal distributions is 0, but numerically this could fail
        # In that case summed_pq_vals_across_sequence can become slightly larger than 1 resulting in NaNs when taking the square root
        return 0.
    L = p_x.shape[0]
    AA = p_x.shape[1]
    summed_pq_vals_across_sequence = []
    for l in range(L):
        alphabet_prod_vals = []
        for a in range(AA):
            pq_sqrt_prod = np.sqrt(p_x[l, a]*q_y[l,a]) # TODO: for weighting: add weighting dist here
            alphabet_prod_vals.append(pq_sqrt_prod)
        summed_alphabet_vals = np.sum(alphabet_prod_vals)
        summed_pq_vals_across_sequence.append(summed_alphabet_vals)
    dist_prod_sum_across_sequence = np.prod(summed_pq_vals_across_sequence)
    assert dist_prod_sum_across_sequence <= 1
    return np.sqrt(1 - dist_prod_sum_across_sequence)


def naive_r_w(p_x: np.ndarray, q_y: np.ndarray, w: np.ndarray):
    """
    p, q are probability distributions (ie. decoder distributions),
    w is weighting distribution (ie. decoder out)
    """
    # assumption sequences x , y are of shape (L, |AA|) with L seq-length and |AA| size of alphabet
    assert p_x.shape[0] == q_y.shape[0] and p_x.shape[1] == q_y.shape[1] and p_x.shape[0] == w.shape[0], "Input distributions inconsistent"
    if np.all(p_x == q_y):
        # the Hellinger distance between equal distributions is 0, but numerically this could fail
        # In that case summed_pq_vals_across_sequence can become slightly larger than 1 resulting in NaNs when taking the square root
        return 0.
    L = p_x.shape[0]
    AA = p_x.shape[1]
    summed_pq_vals_across_sequence = []
    for l in range(L):
        alphabet_prod_vals = []
        for a in range(AA):
            pq_sqrt_prod = w[l,a] * np.sqrt(p_x[l,a]*q_y[l,a])
            alphabet_prod_vals.append(pq_sqrt_prod)
        summed_alphabet_vals = np.sum(alphabet_prod_vals)
        summed_pq_vals_across_sequence.append(summed_alphabet_vals)
    dist_prod_sum_across_sequence = np.prod(summed_pq_vals_across_sequence)
    assert dist_prod_sum_across_sequence <= 1
    lhs_weighted_pq_values = []
    for l in range(L):
        alphabet_prod_vals = []
        for a in range(AA):
            weighted_pq_sum = 1/2 * w[l,a]*p_x[l,a] + 1/2 * w[l,a]*q_y[l,a] 
            alphabet_prod_vals.append(weighted_pq_sum)
        summed_alphabet_vals = np.sum(alphabet_prod_vals)
        lhs_weighted_pq_values.append(summed_alphabet_vals)
    lhs_expectation = np.prod(lhs_weighted_pq_values)
    return np.sqrt(lhs_expectation - dist_prod_sum_across_sequence)


# implement naive Hellinger function
def naive_kernel(p: np.ndarray, q: np.ndarray, theta: float, lam: float) -> float:
    """
    Naive hellinger distance computation and covariance computation of 
    inputs: p , q distribution vectors
        theta, lam covariance function parameters
    returns: 
        kernel value
    """
    distance_mat = np.zeros((p.shape[0], q.shape[0]))
    for i in range(distance_mat.shape[0]):
        for j in range(distance_mat.shape[1]):
            distance_mat[i,j] = naive_r(p[i], q[j])
            if not np.isfinite(distance_mat[i,j]):
                print("Introduced NaN here!")
    return theta * np.exp( -lam * distance_mat)


def naive_weighted_kernel(p: np.ndarray, q: np.ndarray, w: np.ndarray, theta: float, lam: float):
    distance_mat = np.zeros((p.shape[0], q.shape[0]))
    for i in range(distance_mat.shape[0]):
        for j in range(distance_mat.shape[1]):
            distance_mat[i,j] = naive_r_w(p[i], q[j], w)
            if not np.isfinite(distance_mat[i,j]):
                print("NaN here!")
    return theta * np.exp( -lam * distance_mat)


# Simon implementation is p_0 is [N,1]

# def test_kernel_functions_distance_against_naive(): # TODO: this is not the same _hellinger_distance expects vector of 1d not number of elements in alphabet deep!
#     p_0 = simulated_decoding_distributions[:, :, 0] # not weighted p-vec, for comparison against [N, 1], make ones vector of that
#     dist_kernelmodule_function = _hellinger_distance(p_0)
#     dist_naive = naive_r(p_0, p_0)
#     np.testing.assert_almost_equal(dist_kernelmodule_function, dist_naive)


# def test_kernel_functions_k_against_naive_k(): # TODO requires understanding of HD
#     lam = 0.5
#     noise = 0.01
#     module_dist = _hellinger_distance(simulated_decoding_distributions)
## NOTE: Simon's _k is defined over atomic distributions for numerical efficiency, therefore not comparable here
# Test works only on one-hot vectors
#     module_k = _k(module_dist, lengthscale=np.log(lam), log_noise=np.log(noise))
#     naive_k = naive_kernel(simulated_decoding_distributions, simulated_decoding_distributions, theta=1, lam=lam)
#     np.testing.assert_almost_equal(module_k, naive_k)


def test_kernel_implementation_naive():
    """
    test naive sum product reference implementation against GPFlow reference implementation
    """
    theta = 1.
    lam = 1.
    naive_k_matrix = naive_kernel(simulated_decoding_distributions, simulated_decoding_distributions, theta=theta, lam=lam)
    hk = Hellinger(L=L, AA=AA, lengthscale=lam)
    hk_matrix = hk.K(simulated_decoding_distributions, simulated_decoding_distributions)[0].numpy()
    np.testing.assert_allclose(hk_matrix, naive_k_matrix, rtol=1e-6)


def test_weighted_kernel_implementation_naive():
    theta = 1.
    lam = 1.
    whk = WeightedHellinger(w=tf.convert_to_tensor(simulated_weighting_vec), L=L, AA=AA, lengthscale=lam)
    whk_matrix = whk.K(simulated_decoding_distributions, simulated_decoding_distributions)
    naive_whk_matrix = naive_weighted_kernel(simulated_decoding_distributions, simulated_decoding_distributions, simulated_weighting_vec, 
            lam=lam, theta=theta)   
    np.testing.assert_allclose(naive_whk_matrix, whk_matrix[0], 5) # TODO: cov. matrix shape from WHK

# def test_kernel_functions_k():
#     # TODO: test k function in hellinger module
#     assert False

# TODO: test src/corel/kernel GPFlow weighted implementation

# def test_kernel_functions_hd():
#     # TODO: test distance function in hellinger module
#     assert False


if __name__ == "__main__": # NOTE: added for the debugger to work!
    test_kernel_implementation_naive()
    test_weighted_kernel_implementation_naive()