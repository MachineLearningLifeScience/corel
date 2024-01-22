from pathlib import Path

import poli
from poli.core.problem_setup_information import ProblemSetupInformation

from corel.weightings.vae.cbas.cbas_vae_wrapper import CBASVAEWrapper
from corel.weightings.vae.cbas.vae_weighting import VAEWeighting


class CBASVAEFactory:
    def create(self, problem_info: ProblemSetupInformation):
        assert(problem_info.get_problem_name().startswith("gfp_cbas"))
        # NOTE we rely on poli CBas assets to not have duplicate persisted files:
        asset_path = Path(poli.objective_repository.__file__).parent.resolve() / "gfp_cbas"/ "assets" / "models" / "vae"
        return CBASVAEWrapper(AA=len(problem_info.get_alphabet()), L=problem_info.get_max_sequence_length(),
                              prefix=asset_path.resolve().as_posix())

    def get_name(self):
        return self.__class__.__name__


class CBASVAEWeightingFactory:
    def create(self, problem_info: ProblemSetupInformation):
        return VAEWeighting(CBASVAEFactory().create(problem_info))
