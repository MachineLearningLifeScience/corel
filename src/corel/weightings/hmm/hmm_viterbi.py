__author__ = 'Simon Bartels'
import numpy as np


# https://crawlingrobotfortress.blogspot.com/2016/07/python-recipe-for-numerically-stable.html
def viterbi(y, A, B, Pi):
    '''
    See https://en.wikipedia.org/wiki/Viterbi_algorithm

    Parameters
    ----------
    Y : 1D array
        Observations (integer states)
    logP : array shape = (nStates ,)
        1D array of priors for initial state
        given in log probability
    logA : array (nStates,nStates)
        State transition matrix given in log probability
    logB : ndarray K x N
        conditional probability matrix
        log probabilty of each observation given each state
    '''
    #Alpha, c = _forward(Pi, A, B, y)

    logP = np.log(Pi)
    logA = np.log(A)
    logB = np.log(B)

    K = A.shape[0]
    T = len(y)
    logT1 = np.empty((K, T))
    T2 = np.empty((K, T), dtype=np.int64)

    # The initial guess for the first state is initialized as the
    # probability of observing the first observation given said
    # state, multiplied by the prior for that state.
    logT1[:, 0] = logP + logB[:, y[0]]
    # v = logP + logB[:, y[0]]
    # #v -= torch.logsumexp(v - torch.log(torch.tensor(K)), dim=0)
    # #v -= torch.median(v)
    # v -= torch.max(v)
    # #v -= torch.log(c[0])
    # #logT1[:, 0] -= torch.logsumexp(logT1[:, 0], dim=0)
    # Store estimated most likely path
    T2[:, 0] = np.argmax(logT1[:, 0])

    # iterate over all observations from left to right
    for i in range(1, T):
        T2[:, i] = np.argmax(logT1[:, i - 1] + logA.T, axis=1)
        logT1[:, i] = np.max(logT1[:, i - 1] + logA.T, axis=1)
        logT1[:, i] += logB[:, y[i]]
        # v, k = torch.max(v + logA.T, dim=1)
        # v += logB[:, y[i]]  # torch also returns the argmax
        # #logT1[:, i] = v #- torch.log(torch.sum(torch.exp(logT1[:, i-1])))  # renormalize with marginal from previous state?
        # #logT1[:, i] -= torch.log(torch.sum(torch.exp(logT1[:, i])))
        # #v -= torch.logsumexp(v-torch.log(torch.tensor(K)), dim=0)
        # v -= torch.median(v)
        # #v -= torch.max(v)
        # #v -= torch.log(c[i])
        # T2[:, i] = k

    # Build the output, optimal model trajectory
    x = np.empty(T, dtype=np.int64)
    x[-1] = np.argmax(logT1[:, T - 1])
    #x[-1] = torch.argmax(v)
    for i in reversed(range(1, T)):
        x[i - 1] = T2[x[i], i]
    # TODO: remove exp or adapt documentation and test
    return x  #, np.exp(logT1), T2
