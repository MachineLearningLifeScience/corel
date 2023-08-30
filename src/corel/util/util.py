__author__ = 'Simon Bartels'

import numpy as np
import tensorflow as tf
from poli.core.problem_setup_information import ProblemSetupInformation
from gpflow import default_float

from corel.util.constants import PADDING_SYMBOL_INDEX


def get_amino_acid_integer_mapping_from_info(setup_info: ProblemSetupInformation):
    alphabet = setup_info.get_alphabet()
    return {alphabet[i]: i+1 for i in range(len(alphabet))}


def transform_string_sequences_to_integer_arrays(train_x_, L, amino_acid_integer_mapping):
    assert(PADDING_SYMBOL_INDEX not in amino_acid_integer_mapping.values())
    train_x = np.zeros([len(train_x_), L], dtype=int)
    for i in range(len(train_x_)):
        seq = train_x_[i]
        len_seq = len(seq)
        train_x[i, :len_seq] = np.array([amino_acid_integer_mapping[a] for a in seq])
        train_x[i, len_seq:] = PADDING_SYMBOL_INDEX
    return tf.constant(train_x)
