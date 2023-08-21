__author__ = 'Simon Bartels'

import os
from poli.core.problem_setup_information import ProblemSetupInformation

from corel.util.util import get_amino_acid_integer_mapping_from_info
from corel.weightings.hmm.hmm_weighting import HMMWeighting
from hmm_profile import reader


class HMMFactory:
    def __init__(self, hmm_file: str):
        self.hmm_file = hmm_file

    def create(self, problem_info: ProblemSetupInformation):
        amino_acid_integer_mapping = get_amino_acid_integer_mapping_from_info(problem_info)
        # TODO: implement a way to dynamically look for models maybe
        assert(problem_info.get_problem_name() == "FOLDX_RFP")
        #hmm_file = os.path.join(os.path.dirname(__file__), "hmm_models", "rfp.hmm")
        #hmm_file = "/home/simon/stuff/projects/bayesian_optimization/prot_bo/corel/experiments/assets/hmms/rfp.hmm"
        with open(self.hmm_file) as f:
            hmm = reader.read_single(f)
            f.close()
        return HMMWeighting(hmm, amino_acid_integer_mapping)
