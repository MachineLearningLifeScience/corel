__author__ = 'Simon Bartels'

import numpy as np
import tensorflow as tf
from trieste.space import SearchSpaceType, TaggedProductSearchSpace


def make_simplex_optimizer(dataset=None, batch_evaluations=1):
    assert batch_evaluations == 1, "Batch setting currently not supported!"

    def simplex_optimizer(search_space: SearchSpaceType, acquisition_function) -> tf.Tensor:
        assert(type(search_space) is TaggedProductSearchSpace)
        L = search_space.dimension.numpy()
        AA = search_space.upper[0].numpy() + 1
        optimizer = _make_optimizer(L, AA, acquisition_function)
        ps = optimizer()
        x = tf.expand_dims(tf.concat([tf.argmax(p) for p in ps], axis=0), axis=0)
        # TODO: REMOVE
        return tf.cast(x, tf.int32)
    return simplex_optimizer


def _make_optimizer(L, AA, acquisition_function, x0=None):
    if x0 is None:
        x0 = _make_initial_distribution_from_sequence(AA, acquisition_function._model.dataset.query_points[0, :].numpy())
    assert(len(x0.shape) == 2)
    assert(x0.shape[0] == L)
    assert(x0.shape[1] == AA)
    xs = [tf.Variable(x0[i:i+1, :].T) for i in range(L)]

    #@tf.function(reduce_retracing=True)
    @tf.function
    def _optimize():
        # TODO: optimize as long as gradient is too large!
        grad_norm = tf.float64.max
        it = 1
        while grad_norm > 0.:  #1e-15:
            tf.print("simplex gradient descent step " + str(it))
            # TODO: use persistent flag?
            with tf.GradientTape() as t:
                x = tf.expand_dims(tf.transpose(tf.concat(xs, axis=0)), axis=0)
                a = acquisition_function(x)
            ys = t.gradient(a, xs)
            # tf.print(ys[0])
            # with tf.GradientTape(persistent=True) as t:
            #     x = tf.transpose(tf.concat(xs, axis=0))
            #     m, v = acquisition_function._model.predict(x)
            # print(t.gradient(m, x))
            # print(t.gradient(v, x))
            # exit()
            # break

            # if not tf.reduce_all(tf.math.is_finite(ys)):
            #     # try finite differences gradient
            #     epsilon = 1e-8
            #     ys = []
            #     for i in range(L):
            #         for j in range(AA):
            #             ym = acquisition_function(tf.expand_dims(tf.transpose(x-epsilon), axis=0))
            #             yp = acquisition_function(tf.expand_dims(tf.transpose(x+epsilon), axis=0))
            #     ys = [tf.constant(yp[i*AA:(i+1)*AA] - ym[i*AA:(i+1)*AA]) / 2 / epsilon for i in range(L)]
            # gradient formula taken from here:
            # https://docs.juliahub.com/Manifolds/H884l/0.8.60/autodocs/#ManifoldDiff.riemannian_gradient-Tuple{ProbabilitySimplex,%20Any,%20Any
            #gs = [ys[i] * xs[i] - tf.transpose(xs[i]) @ ys[i] * xs[i] for i in range(L)]
            grad_norm = 0
            alpha = _linesearch(it, acquisition_function, xs, gs=None)
            argmax_is_optimum = True
            for i in range(L):
                gsi = ys[i] * xs[i] - tf.transpose(xs[i]) @ ys[i] * xs[i]
                if tf.reduce_all(tf.abs(gsi) == 0.):
                    continue
                strongest_push = tf.argmax(gsi).numpy()[0]
                most_likely_amino_acid = tf.argmax(xs[i]).numpy()[0]
                most_likely_amino_acid_is_best = strongest_push == most_likely_amino_acid
                beta = 1.
                if not most_likely_amino_acid_is_best:
                    grad_diff = max(1e-15, gsi[strongest_push] - gsi[most_likely_amino_acid])  # avoid division by 0
                    beta = 1. / alpha * (xs[i][most_likely_amino_acid] - xs[i][strongest_push] + 1e-15) / grad_diff
                    raise NotImplementedError("TODO: the padding symbol index is creating problems here")
                    longest_possible_move = -tf.reduce_min(xs[i] / gsi)
                    beta = min(longest_possible_move / alpha, beta)

                argmax_is_optimum = argmax_is_optimum and most_likely_amino_acid_is_best
                grad_norm += tf.square(tf.norm(gsi))
                # negative sign maximizes, which we want for the acquisition function
                # TODO: do I want a line search?
                xs[i].assign(xs[i].value() + alpha * beta * gsi)
                # gsi.T @ 1 ~= 0, but the error can accumulate. It's important to keep on renormalizing
                xs[i].assign(xs[i] / tf.reduce_sum(xs[i]))
            grad_norm = tf.sqrt(alpha * grad_norm)
            it += 1
            if argmax_is_optimum:
                break
        return xs
    return _optimize


def _make_initial_distribution_from_sequence(AA: int, seq: np.ndarray) -> np.ndarray:
    assert(len(seq.shape) == 1)
    L = seq.shape[0]
    # x0 = tf.expand_dims(search_space.lower, axis=0)
    #normalize = lambda x: x / tf.reduce_sum(x)
    # xs = [tf.Variable(normalize(tf.random.uniform([AA, 1], dtype=default_float()))) for _ in range(L)]
    # xs = [tf.Variable(normalize(tf.ones([AA, 1], dtype=default_float()))) for _ in range(L)]
    epsilon = 1e-8
    one_minus_Z = 1. - (AA - 1) * epsilon
    x0 = epsilon * np.ones([L, AA])
    x0[np.arange(L), seq] = one_minus_Z
    return x0


def _linesearch(it, acquisition_function, xs, gs=None):
    return 1. / it
