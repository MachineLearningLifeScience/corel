__author__ = 'Richard Michael'

from poli.core.problem_setup_information import ProblemSetupInformation

from corel.util.util import get_amino_acid_integer_mapping_from_info

from vae import VAE


class VAEFactory:
    def __init__(self, vae_file: str, problem_name: str, **kwargs):
        """
        :Input:
            vae_file: str - specifies latest checkpoint file (tf) of fitted model,
            problem_name: str - problem identifier for poli Problem Information lookup
            kwargs: dict - specify VAE attributes as dict
        """
        self.vae_file = vae_file
        self.problem_name = problem_name
        self.kwargs = kwargs
        # NOTE: specify VAE architecture in kwargs
    # TODO: implement call function

    def create(self, problem_info: ProblemSetupInformation):
        # the assertion below is a way to allow the problem mapping to be safely part of the experiment and keeping it out of the corel package
        #assert(problem_info.get_problem_name() == self.problem_name)
        # TODO: assess if AA mapping and/or problem_info are required here for later purposes
        # amino_acid_integer_mapping = get_amino_acid_integer_mapping_from_info(problem_info)
        vae = VAE(
            z_dim=self.kwargs.get("z_dim"),
            input_dims=self.kwargs.get("input_dims"),
            n_categories=self.kwargs.get("n_categories"),
            encoder_layers=self.kwargs.get("encoder_layers"),
            decoder_layers=self.kwargs.get("decoder_layers"),
            )
        vae.model.load_weights(self.vae_file).expect_partial()
        return vae

    def get_name(self):
        return self.__class__.__name__
