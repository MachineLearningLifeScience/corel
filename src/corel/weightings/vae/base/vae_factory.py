__author__ = 'Richard Michael'
import tensorflow as tf

from poli.core.problem_setup_information import ProblemSetupInformation

from corel.util.util import get_amino_acid_integer_mapping_from_info

from corel.weightings.vae.base import VAE
from corel.weightings.vae.base.train_vae import _preprocess

# TODO: this is bad practice, ADD experiments module to setup cfg
import sys
if "/Users/rcml/corel/" not in sys.path:
    sys.path.append("/Users/rcml/corel/")
from experiments.assets.data.rfp_fam import rfp_train_dataset, rfp_test_dataset
train_dataset = rfp_train_dataset.map(_preprocess).batch(128).prefetch(tf.data.AUTOTUNE).shuffle(42)
test_dataset = rfp_test_dataset.map(_preprocess).batch(128).prefetch(tf.data.AUTOTUNE)
x0 = next(iter(train_dataset))[0][0]

base_vae_config = dict( # TODO: make this part of a central config: use for training, instantiating factory function etc.
    z_dim=2, # =10 
    input_dims=x0.shape,
    n_categories=tf.constant(20), # NOTE: see tf.constant(len(AMINO_ACIDS))
    encoder_layers=[1000, 250], 
    decoder_layers=[250, 1000],
)


class VAEFactory:
    def __init__(self, vae_file: str, problem_name: str, vae_config=base_vae_config, **kwargs):
        """
        :Input:
            vae_file: str - specifies latest checkpoint file (tf) of fitted model,
            problem_name: str - problem identifier for poli Problem Information lookup
            kwargs: dict - specify VAE attributes as dict
        """
        self.vae_file = vae_file
        self.problem_name = problem_name
        self.vae_config=vae_config
        self.kwargs = kwargs # NOTE: specify VAE architecture in kwargs
    # TODO: implement call function

    def create(self, problem_info: ProblemSetupInformation):
        # the assertion below is a way to allow the problem mapping to be safely part of the experiment and keeping it out of the corel package
        #assert(problem_info.get_problem_name() == self.problem_name)
        # TODO: assess if AA mapping and/or problem_info are required here for later purposes
        # amino_acid_integer_mapping = get_amino_acid_integer_mapping_from_info(problem_info)
        vae = VAE(
            z_dim=self.vae_config.get("z_dim"),
            input_dims=self.vae_config.get("input_dims"),
            n_categories=self.vae_config.get("n_categories"),
            encoder_layers=self.vae_config.get("encoder_layers"),
            decoder_layers=self.vae_config.get("decoder_layers"),
            )
        vae.model.load_weights(self.vae_file).expect_partial()
        return vae

    def get_name(self):
        return self.__class__.__name__
