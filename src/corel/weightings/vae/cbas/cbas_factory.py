from pathlib import Path

import poli
from poli.core.problem_setup_information import ProblemSetupInformation

from corel.weightings.vae.cbas.cbas_vae_wrapper import CBASVAEWrapper
from corel.weightings.vae.cbas.vae_weighting import VAEWeighting


class CBASVAEFactory:
    def create(self, problem_info: ProblemSetupInformation, asset_path: Path=None, latent_dim=20):
        assert problem_info.get_problem_name().startswith("gfp_cbas"), "Incorrect Problem for CBas weighting"
        # NOTE we rely on poli CBas assets to not have duplicate persisted files:
        if asset_path is None:
            asset_path = Path(poli.objective_repository.__file__).parent.resolve() / "gfp_cbas"/ "assets" / "models" / "vae"
        return CBASVAEWrapper(AA=len(problem_info.get_alphabet()), L=problem_info.get_max_sequence_length(),
                              prefix=asset_path.resolve().as_posix(), latent_dim=latent_dim)

    def get_name(self):
        return self.__class__.__name__


class CBASVAEWeightingFactory:
    def create(self, problem_info: ProblemSetupInformation, model_path: Path=None, latent_dim: int=20):
        return VAEWeighting(CBASVAEFactory().create(problem_info, asset_path=model_path, latent_dim=latent_dim))
