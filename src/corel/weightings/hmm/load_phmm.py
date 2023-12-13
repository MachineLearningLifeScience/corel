import logging
from hmm_profile import reader
import numpy as np


TERMINATION_SYMBOL = 'x'
ALPHABET_KEY = 'alphabet'


def load_hmm(model):
    return transform_phmm_to_hmm(*load_phmm(model))


def transform_phmm_to_hmm(s0, T, em, extra_info_dict):
    N = (T.shape[0] - 2) // 3 * 2 + 2  # number of states
    assert((N - 2) / 2 * 3 + 2 == T.shape[0])
    M = (N - 2) // 2  # number of match states
    T_ = np.zeros([N, N])
    em_ = np.zeros([N, em.shape[1]])

    # we will keep m0 as unreachable state

    em_[0, :] = em[0, :]
    #T[0, 0] = model.start_step.p_insertion_to_insertion  # i0->i0
    T_[0, 0] = T[0, 0]
    #T[0, 2] = model.start_step.p_insertion_to_emission  # i0->m1
    T_[0, 2] = T[0, 2]

    for s in range(M-1):
        #T[s*3+1, s*3+1] = step.p_insertion_to_insertion  # i->i
        T_[s*2+1, s*2+1] = T[s*3+1, s*3+1]
        #T[s*3+1, (s+1)*3+2] = step.p_insertion_to_emission  # i->m+1
        T_[s*2+1, (s+1)*2+2] = T[s*3+1, (s+1)*3+2]

        #T[s*3+2, s*3+1] = step.p_emission_to_insertion  # m->i
        T_[s*2+2, s*2+1] = T[s*3+2, s*3+1]
        #T[s*3+2, (s+1)*3+2] = step.p_emission_to_emission  # m->m+1
        T_[s*2+2, (s+1)*2+2] = T[s*3+2, (s+1)*3+2]
        em_[s * 2 + 1, :] = em[s * 3 + 1, :]
        em_[s * 2 + 2, :] = em[s * 3 + 2, :]

        #deletion_prob = 1.
        deletion_path = [T[i*3+3, (i+1)*3+3] for i in range(s-1)]
        for j in range(s-1):
        #for j in range(s-2, -1, -1):
            # only match states are connected by delete states
            # T[s*3+2, (s+1)*3+3] = step.p_emission_to_deletion  # m->d+1
            # TODO: maybe a bit inefficient
            #deletion_path = [T[i*3+3, (i+1)*3+3] for i in range(j+1, s-1)]
            T_[j*2+2, s*2+2] = T[j*3+2, (j+1)*3+3] * np.prod(deletion_path[j+1:]) * T[(s-1)*3+3, s*3+2]

    s = M - 1
    #T[s*3+1, s*3+1] = step.p_insertion_to_insertion  # i->i
    T_[s*2+1, s*2+1] = T[s*3+1, s*3+1]
    #T[s*3+1, (s+1)*3+2] = step.p_insertion_to_emission  # i->m+1
    T_[s*2+1, -1] = T[s*3+1, -1]

    #T[s*3+2, s*3+1] = step.p_emission_to_insertion  # m->i
    T_[s*2+2, s*2+1] = T[s*3+2, s*3+1]
    #T[s*3+2, (s+1)*3+2] = step.p_emission_to_emission  # m->m+1
    T_[s*2+2, -1] = T[s*3+2, -1]
    em_[s * 2 + 1, :] = em[s * 3 + 1, :]
    em_[s * 2 + 2, :] = em[s * 3 + 2, :]

    end_state_index = -1  #s*2+2  # actually, last "real" match state index
    for j in range(s-1):
        # only match states are connected by delete states
        # T[s*3+2, (s+1)*3+3] = step.p_emission_to_deletion  # m->d+1
        # TODO: maybe a bit inefficient
        deletion_path = [T[i*3+3, (i+1)*3+3] for i in range(j+1, s-1)]
        T_[j*2+2, s*2+2] = T[j*3+2, (j+1)*3+3] * np.prod(deletion_path) * T[(s-1)*3+3, s*3+2]
        T_[j*2+2, end_state_index] = T[j*3+2, (j+1)*3+3] * np.prod(deletion_path) * T[(s-1)*3+3, s*3+3] #* T[s*3+3, -1]  # this last entry is just one
    j = s - 1
    T_[j * 2 + 2, end_state_index] = T[j*3+2, s*3+3] #+ T[j*3+2, s*3+2]

    T_[-1, -1] = 1.
    em_[-1, -1] = 1.

    # # process last step
    # s = M - 1
    # end_state_index = N-1
    # #em_[end_state_index, -1] = 1  # the end state only produces the eot symbol
    # T_[s*2+1, s*2+1] = T[s*3+1, s*3+1]  # i->i
    # T_[s*2+1, end_state_index] = T[s*3+1, -1]  # jump to end state
    # T_[s*2+2, s*2+1] = T[s*3+2, s*3+1]  # this is still m->i
    # T_[s*2+2, end_state_index] = T[s*3+2, -1]  # jump to end state
    # #j = s-1
    # #T_[j * 2 + 2, s * 2 + 2] = T[j * 3 + 2, (j + 1) * 3 + 3] * np.prod(deletion_path[j + 1:]) * T[(s - 1) * 3 + 3, s * 3 + 2]
    # #T[s*3+2, (s+1)*3+3] = step.p_emission_to_deletion
    # #assert(step.p_deletion_to_emission == 1.)
    # #T[s*3+3, end_state_index] = step.p_deletion_to_emission  # jump to end state
    # #T[(s+1)*3+1, (s+1)*3+1] = 1.  # making the E state going to itself?
    # T_[end_state_index, end_state_index] = 1.
    # em_[s * 2 + 1, :] = em[s * 3 + 1, :]
    # em_[s * 2 + 2, :] = em[s * 3 + 2, :]
    # for j in range(s-1):
    #     # only match states are connected by delete states
    #     # T[s*3+2, (s+1)*3+3] = step.p_emission_to_deletion  # m->d+1
    #     # TODO: maybe a bit inefficient
    #     deletion_path = [T[i*3+3, (i+1)*3+3] for i in range(j+1, s-1)]
    #     T_[j*2+2, s*2+2] = T[j*3+2, (j+1)*3+3] * np.prod(deletion_path) * T[(s-1)*3+3, s*3+2]
    #     T_[j*2+2, end_state_index] = T[j*3+2, (j+1)*3+3] * np.prod(deletion_path) * T[(s-1)*3+3, s*3+3] #* T[s*3+3, -1]  # this last entry is just one
    # j = s - 1
    # T_[j * 2 + 2, end_state_index] = T[j*3+2, s*3+3]

    s0_ = np.zeros(N)
    s0_[0] = s0[0]  # i0
    assert(s0[1] == 0.)
    s0_[2] = s0[2]  # m1
    deletion_path = [T[i * 3 + 3, (i + 1) * 3 + 3] for i in range(M - 1)]
    for j in range(1, M):
        s0_[2*j+2] = s0[3] * np.prod(deletion_path[:j-1]) * T[3*(j-1)+3, 3*j+2]
    #s0[3] = model.start_step.p_emission_to_deletion

    assert(s0_[1] == 0.)  # i1 is only reachable from m1
    T_[-1, -1] = 1.
    em_[-1, -1] = 1.
    try:
        np.testing.assert_almost_equal(np.sum(s0_), np.ones(1), decimal=5)
        np.testing.assert_almost_equal(np.sum(em_, axis=1), np.ones(N), decimal=5)
        np.testing.assert_almost_equal(np.sum(T_, axis=1), np.ones(N), decimal=5)
    except Exception as e:
        logging.exception(e)
        raise e
    T_ = np.diag(1. / np.sum(T_, axis=1)) @ T_  # make proper transition matrix to correct for numerical instabilities
    em_ = np.diag(1. / np.sum(em_, axis=1)) @ em_
    return s0_, T_, em_, extra_info_dict


def load_phmm_from_file(file_name='pf00144.hmm'):
    with open(file_name) as f:
        model = reader.read_single(f)
        f.close()
        return model


def load_phmm(model):
    #L = model.metadata.length  # this is the number of match states...
    #L = model.steps[-1].alignment_column_index
    #N = model.metadata.length * 3 + 3  # silent m0, i0 and m->m, m->i, m->d, i->m, i->i, d->m, d->d
    N = model.metadata.length * 3 + 2  # i0, then (i, m, d) * N, finally E
    # the silent starting state m0 is taken care of by s0
    O = len(model.metadata.alphabet) + 1
    mapping = model.metadata.alphabet
    #del_sym_pos = 0
    #mapping.insert(del_sym_pos, '-')

    # TODO: to make the update to the backward pass I could move all silent states into one part of the transition matrix
    # Then I do not need to store an extra emission matrix for the 2nd part of the backward pass
    # I can even cut down the emission probabilities for the silent states
    # WOW, this means this is a kernel for unaligned sequences!

    s0 = np.zeros(N)
    s0[0] = model.start_step.p_emission_to_insertion  # i0
    # i1 has 0 chance of being starting state
    s0[2] = model.start_step.p_emission_to_emission  # m1
    s0[3] = model.start_step.p_emission_to_deletion  # d1
    np.testing.assert_almost_equal(np.sum(s0), np.ones(1), decimal=5)
    em = np.zeros((N, O))  # ones for the silent states
    for i in range(O-1):
        em[0, i] = model.start_step.p_insertion_char[mapping[i]]
    T = np.zeros((N, N))

    # # this state (m0) is silent!
    # # we will store this begin state as last state!
    # T[-1, 0] = model.start_step.p_emission_to_insertion
    # T[-1, 2] = model.start_step.p_emission_to_emission
    # T[-1, 3] = model.start_step.p_emission_to_deletion

    T[0, 0] = model.start_step.p_insertion_to_insertion  # i0->i0
    T[0, 2] = model.start_step.p_insertion_to_emission  # i0->m1
    for s in range(model.metadata.length - 1):
        step = model.steps[s]
        T[s*3+1, s*3+1] = step.p_insertion_to_insertion  # i->i
        T[s*3+1, (s+1)*3+2] = step.p_insertion_to_emission  # i->m+1

        T[s*3+2, s*3+1] = step.p_emission_to_insertion  # m->i
        T[s*3+2, (s+1)*3+2] = step.p_emission_to_emission  # m->m+1
        T[s*3+2, (s+1)*3+3] = step.p_emission_to_deletion  # m->d+1

        T[s*3+3, (s+1)*3+2] = step.p_deletion_to_emission  # d->m+1
        T[s*3+3, (s+1)*3+3] = step.p_deletion_to_deletion  # d->d+1
        for i in range(O-1):
            em[s*3+1, i] = step.p_insertion_char[mapping[i]]
            em[s*3+2, i] = step.p_emission_char[mapping[i]]
        #em[s*3+3, del_sym_pos] = 1.

    # process last step
    s = model.metadata.length - 1
    step = model.steps[s]
    end_state_index = -1  #s*3+2  # by convention: last match state
    #assert(end_state_index != N-1)
    #T[end_state_index, end_state_index] = 1.
    T[s*3+1, s*3+1] = step.p_insertion_to_insertion
    T[s*3+1, end_state_index] = step.p_insertion_to_emission  # jump to end state
    T[s*3+2, s*3+1] = step.p_emission_to_insertion  # this is still m->i
    T[s*3+2, end_state_index] = step.p_emission_to_emission  # jump to end state
    #T[s*3+2, (s+1)*3+3] = step.p_emission_to_deletion
    assert(step.p_deletion_to_emission == 1.)
    T[s*3+3, end_state_index] = step.p_deletion_to_emission  # jump to end state
    #T[(s+1)*3+1, (s+1)*3+1] = 1.  # making the E state going to itself?
    for i in range(O-1):
        em[s * 3 + 1, i] = step.p_insertion_char[mapping[i]]
        em[s * 3 + 2, i] = step.p_emission_char[mapping[i]]
    #em[s * 3 + 3, del_sym_pos] = 1.
    #em[(s+1)*3+1, del_sym_pos] = 1.  # END state only produces deletion characters?
    em[end_state_index, -1] = 1.
    T[end_state_index, end_state_index] = 1.

    try:
        np.testing.assert_almost_equal(np.sum(T, axis=1), np.ones(N), decimal=5)
        # we don't check emissions here but above because we don't want to include the delete states
        #np.testing.assert_almost_equal(np.sum(em, axis=1), np.ones(N), decimal=5)
        np.testing.assert_almost_equal(np.sum(s0), np.ones(1), decimal=5)
    except Exception as e:
        logging.exception(e)
        raise e
    return s0, T, em, {ALPHABET_KEY: mapping + [TERMINATION_SYMBOL]}
