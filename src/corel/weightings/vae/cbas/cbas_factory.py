from pathlib import Path

from poli.core.problem_setup_information import ProblemSetupInformation

from corel.weightings.vae.cbas.cbas_vae_wrapper import CBASVAEWrapper
from corel.weightings.vae.cbas.vae_weighting import VAEWeighting


class CBASVAEFactory:
    def create(self, problem_info: ProblemSetupInformation):
        assert(problem_info.get_problem_name() == "gfp_cbas")
        return CBASVAEWrapper(AA=len(problem_info.get_alphabet()), L=problem_info.get_max_sequence_length(),
                              prefix=Path("./assets/vaes/gfp").resolve().as_posix())

    def get_name(self):
        return self.__class__.__name__


class CBASVAEWeightingFactory:
    def create(self, problem_info: ProblemSetupInformation):
        return VAEWeighting(CBASVAEFactory().create(problem_info))
