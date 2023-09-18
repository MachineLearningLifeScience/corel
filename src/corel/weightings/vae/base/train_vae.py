import tensorflow as tf
from warnings import warn
import os

# TODO: this is bad practice, ADD experiments module to setup cfg
import sys
if "/Users/rcml/corel/" not in sys.path:
    sys.path.append("/Users/rcml/corel/")
from experiments.assets.data.rfp_fam import rfp_train_dataset, rfp_test_dataset
from experiments.assets.data.blat_fam import blat_train_dataset, blat_test_dataset
from corel.weightings.vae.base.models import VAE
from corel.weightings.vae.base import LATENT_DIM


# TODO: make this alphabet part of a ProblemInfo
AMINO_ACIDS = [
            "A",
            "R",
            "N",
            "D",
            "C",
            "E",
            "Q",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
            "-",
        ]


def _preprocess(x, vocab=AMINO_ACIDS):
    tokenized_string = tf.strings.bytes_split(x)
    aa_oh_lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode="one_hot")
    _x = aa_oh_lookup(tokenized_string)
    return _x, _x


if __name__ == "__main__":
    BATCHSIZE = 200 #128
    EPOCHS = 100 # RFP=1000 BLAT = 100
    SEED = 42
    LR = 1e-3
    cpu = False
    dataset = "blat_fam" # "rfp_fam" #
    if tf.test.gpu_device_name() != "/device:GPU:0":
        cpu = True
        warn("GPU device not found.")
    else:
        print(f"SUCCESS: Found GPU: {tf.test.gpu_device_name()}")

    if dataset == "rfp_fam":
        train_dataset = rfp_train_dataset.map(_preprocess).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE).shuffle(SEED)
        eval_dataset = rfp_test_dataset.map(_preprocess).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)
    elif dataset == "blat_fam":
        train_dataset = blat_train_dataset.map(_preprocess).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE).shuffle(SEED)
        eval_dataset = blat_test_dataset.map(_preprocess).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)
    elif dataset == "zinc":
        raise NotImplementedError("TODO!")
    else:
        raise ValueError("Specify dataset from [RFP ; BLAT]!")

    x0 = next(iter(train_dataset))[0][0]

    vae = VAE(z_dim=LATENT_DIM, input_dims=x0.shape, n_categories=x0.shape[-1])

    if cpu: # NOTE: M1/M2 processors require legacy Adam
        optimizer = tf.optimizers.legacy.Adam(learning_rate=LR)
    else:
        optimizer = tf.optimizers.Adam(learning_rate=LR)

    MODEL_PATH = f"results/models/vae_z_{vae.encoder.z_dim}_{dataset}.ckpt"
    checkpoint_dir = os.path.dirname(MODEL_PATH)
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH,
                                                    save_weights_only=True,
                                                    verbose=1)

    vae.model.compile(optimizer=optimizer, loss=lambda x, model: -model.log_prob(x))

    _ = vae.model.fit(train_dataset, epochs=EPOCHS, validation_data=eval_dataset, callbacks=[cp_callback])