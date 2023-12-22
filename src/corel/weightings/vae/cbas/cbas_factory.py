from poli.core.problem_setup_information import ProblemSetupInformation

from corel.weightings.vae.cbas.cbas_vae_wrapper import CBASVAEWrapper


class CBASVAEFactory:
    def create(self, problem_info: ProblemSetupInformation):
        return CBASVAEWrapper(AA=len(problem_info.get_alphabet()), L=problem_info.get_max_sequence_length(),
                              prefix="experiments/assets/vaes/gfp")

    def get_name(self):
        return self.__class__.__name__
