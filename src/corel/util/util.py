__author__ = 'Simon Bartels'

import tensorflow as tf
from gpflow import default_float
from poli.core.problem_setup_information import ProblemSetupInformation


def get_amino_acid_integer_mapping_from_info(setup_info: ProblemSetupInformation):
    alphabet = setup_info.get_alphabet()
    return {alphabet[i]: i+1 for i in range(len(alphabet))}


def seq_to_atom(x):
    raise NotImplementedError("get alphabet size!")
    return tf.reshape(tf.one_hot(x, depth=20, axis=-1, dtype=default_float()), [x.shape[0], 1, 237*20])
