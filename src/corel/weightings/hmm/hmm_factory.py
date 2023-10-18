__author__ = 'Simon Bartels'

import os
from poli.core.problem_setup_information import ProblemSetupInformation

from corel.util.util import get_amino_acid_integer_mapping_from_info
from corel.weightings.hmm.hmm_weighting import HMMWeighting
from hmm_profile import reader
from corel.weightings.hmm.load_phmm import load_hmm


class HMMFactory:
    def __init__(self, hmm_file: str, problem_name: str):
        self.hmm_file = hmm_file
        self.problem_name = problem_name

    def create(self, problem_info: ProblemSetupInformation):
        # the assertion below is a way to allow the problem mapping to be safely part of the experiment and keeping it out of the corel package
        assert(problem_info.get_problem_name() == self.problem_name)
        amino_acid_integer_mapping = get_amino_acid_integer_mapping_from_info(problem_info)
        with open(self.hmm_file) as f:
            hmm = reader.read_single(f)
            f.close()
        s0, T, em, extra_info_dict = load_hmm(hmm)
        hmm_alphabet = hmm.metadata.alphabet
        return HMMWeighting(s0, T, em, hmm_alphabet, amino_acid_integer_mapping)

    def get_name(self):
        return self.__class__.__name__
