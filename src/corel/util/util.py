__author__ = 'Simon Bartels'

import numpy as np
import tensorflow as tf
from poli.core.problem_setup_information import ProblemSetupInformation
from gpflow import default_float

from corel.util.constants import PADDING_SYMBOL_INDEX


def get_amino_acid_integer_mapping_from_info(setup_info: ProblemSetupInformation):
    alphabet = setup_info.get_alphabet()
    offset = 0 if setup_info.sequences_are_aligned() else 1
    assert(PADDING_SYMBOL_INDEX == 0)
    return {alphabet[i]: i+offset for i in range(len(alphabet))}


def transform_string_sequences_to_integer_arrays(train_x_, L, amino_acid_integer_mapping):
    # the assertion below is only valid if sequences are not aligned
    #assert(PADDING_SYMBOL_INDEX not in amino_acid_integer_mapping.values())
    train_x = np.zeros([len(train_x_), L], dtype=int)
    assert(PADDING_SYMBOL_INDEX == 0)
    for i in range(len(train_x_)):
        seq = train_x_[i]
        len_seq = len(seq)
        train_x[i, :len_seq] = np.array([amino_acid_integer_mapping[a] for a in seq])
        train_x[i, len_seq:] = PADDING_SYMBOL_INDEX
    return tf.constant(train_x)


def handle_batch_shape(X: tf.Tensor) -> tf.Tensor:
    if len(X.shape) == 4:
        if X.shape[0] == 1:
            X = tf.reshape(X, shape=X.shape[1:])
        elif X.shape[1] == 1:
            X = tf.reshape(X, shape=(X.shape[0], X.shape[2]))
        else:
            raise NotImplementedError(f"{X.shape}")
    return X
