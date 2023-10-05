from warnings import warn
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

from corel.util.small_molecules.data import load_zinc_250k_dataset
from corel.weightings.vae.small_molecules.vae_selfies import VAESelfies
from corel.weightings.vae.base import LATENT_DIM

# tf.config.run_functions_eagerly(True)

ENCODING_LAYERS = [1024, 512, 256, 128]
DECODING_LAYERS = [128, 256, 512, 1024]


def _preprocess(x):
    """
    The model is expecting flat one-hot inputs, and
    integer labels as outp.
    """
    return x, x


if __name__ == "__main__":
    # Defining some hyperparameters
    BATCHSIZE = 256  # 128
    EPOCHS = 300  # RFP=1000 BLAT = 100
    SEED = np.random.randint(100)
    LR = 1e-3
    cpu = False

    # loading up the dataset
    all_onehot_arrays = load_zinc_250k_dataset()
    dataset_size, sequence_length, n_categories = all_onehot_arrays.shape
    all_onehot_arrays = tf.constant(
        all_onehot_arrays,
        dtype=tf.float32,
        name="all_onehot_arrays",
    )

    # Shuffling before splitting
    all_onehot_arrays = tf.random.shuffle(all_onehot_arrays, seed=SEED)

    # TEST: let's see if we can overfit to a couple of batches
    all_onehot_arrays = all_onehot_arrays[:2 * BATCHSIZE]

    # Splitting into train and test (80/20)
    train_size = int(0.8 * len(all_onehot_arrays))
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(all_onehot_arrays[:train_size])
        .map(_preprocess)
        .batch(BATCHSIZE)
        .shuffle(SEED)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(all_onehot_arrays[train_size:])
        .map(_preprocess)
        .batch(BATCHSIZE)
        .shuffle(SEED)
    )

    if tf.test.gpu_device_name() != "/device:GPU:0":
        cpu = True
        warn("GPU device not found.")
    else:
        print(f"SUCCESS: Found GPU: {tf.test.gpu_device_name()}")

    vae = VAESelfies(
        z_dim=LATENT_DIM,
        input_dims=(sequence_length, n_categories),
        n_categories=n_categories,
        encoder_layers=ENCODING_LAYERS,
        decoder_layers=DECODING_LAYERS,
    )

    if cpu:  # NOTE: M1/M2 processors require legacy Adam
        optimizer = tf.optimizers.legacy.Adam(learning_rate=LR)
    else:
        optimizer = tf.optimizers.Adam(learning_rate=LR)

    MODEL_PATH = f"results/models/vae_z_{vae.encoder.z_dim}_zinc250k.ckpt"
    checkpoint_dir = os.path.dirname(MODEL_PATH)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=MODEL_PATH, save_weights_only=True, verbose=1
    )

    vae.model.compile(
        optimizer=optimizer,
        loss=lambda x, model: -model.log_prob(x),
    )

    _ = vae.model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
        callbacks=[cp_callback],
    )
