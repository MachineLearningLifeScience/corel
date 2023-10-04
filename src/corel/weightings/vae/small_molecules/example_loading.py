"""This script contains an example of how to load a trained the VAESelfies

Using one of the trained models from the `results/models` directory, we load up the model and use it to generate new molecules.
"""
from pathlib import Path

import tensorflow as tf

from corel.weightings.vae.small_molecules.vae_selfies import VAESelfies
from corel.weightings.vae.base import LATENT_DIM
from corel.util.small_molecules.data import load_zinc_250k_dataset

# Defining the path for the weights

if __name__ == "__main__":
    # Getting the sequence length and number of categories
    _, sequence_length, n_categories = load_zinc_250k_dataset().shape

    # Creating an instance of the model
    vae = VAESelfies(
        z_dim=LATENT_DIM,
        input_dims=(sequence_length * n_categories),
        n_categories=n_categories,
    )

    # Defining the path to the weights
    ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent.resolve()
    weights_path = ROOT_DIR / "results" / "models" / "vae_z_2_zinc250k.ckpt"

    # Loading the weights
    vae.model.load_weights(weights_path)

    # Generating a new molecule
    latent_code = tf.random.normal(shape=(1, LATENT_DIM))

    cat_distribution = vae.decoder.layers(latent_code)
    print(cat_distribution)
