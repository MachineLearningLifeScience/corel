from pathlib import Path

from corel.weightings.vae.small_molecules.train_vae import (
    ENCODING_LAYERS,
    DECODING_LAYERS,
    LATENT_DIM,
    VAESelfies,
)
from corel.util.small_molecules.data import load_zinc_250k_dataset


def load_vae() -> VAESelfies:
    # Getting the sequence length and number of categories
    _, sequence_length, n_categories = load_zinc_250k_dataset().shape

    # Creating an instance of the model
    vae = VAESelfies(
        z_dim=LATENT_DIM,
        input_dims=(sequence_length, n_categories),
        n_categories=n_categories,
        encoder_layers=ENCODING_LAYERS,
        decoder_layers=DECODING_LAYERS,
    )

    # Defining the path to the weights
    ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()
    weights_path = ROOT_DIR / "results" / "models" / "vae_z_2_zinc250k.ckpt"

    # Loading the weights
    vae.model.load_weights(weights_path)

    return vae
