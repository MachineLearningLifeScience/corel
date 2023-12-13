"""Testing whether o.o.d inputs are highly entropic.

In this experiment, we load the VAESelfies and pass a
random sequence of length 70 through it. We then
calculate the entropy of the output, and compare it
to the entropy of the training set. We also visualize
entropy in the latent space.
"""
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

from corel.weightings.vae.small_molecules.load_vae import load_vae
from corel.util.small_molecules.data import (
    load_zinc_250k_dataset,
    load_zinc_250k_alphabet,
)


def passing_a_random_selfies_through():
    max_sequence_length = 70
    n_inputs = 3
    alphabet = load_zinc_250k_alphabet()
    vae = load_vae()

    random_input_as_ints = np.random.randint(
        0, len(alphabet), size=(n_inputs, max_sequence_length)
    )
    random_input_as_onehot = np.zeros((n_inputs, max_sequence_length, len(alphabet)))
    for i, row in enumerate(random_input_as_ints):
        for j, col in enumerate(row):
            random_input_as_onehot[i, j, col] = 1

    random_input_as_onehot = tf.constant(random_input_as_onehot, dtype=tf.float32)
    latent_dist = vae.encoder.layers(random_input_as_onehot)
    latent_mean = latent_dist.mean()

    print(f"Latent mean: {latent_mean}")
    dist_for_random_input = vae.decoder.layers(latent_mean)
    entropy = dist_for_random_input.entropy()

    _, axes = plt.subplots(1, n_inputs)
    for entropy_i, ax in zip(entropy, axes):
        ax.bar(np.arange(len(entropy_i)), entropy_i)

    print(f"Example input: {random_input_as_ints[0]}")
    print(
        f"Respective reconstruction: {dist_for_random_input.logits.numpy().argmax(axis=-1)[0]}"
    )

    plt.show()
    ...


if __name__ == "__main__":
    passing_a_random_selfies_through()
    ...
