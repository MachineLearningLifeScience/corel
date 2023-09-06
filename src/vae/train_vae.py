import tensorflow as tf
from warnings import warn
import os

# TODO: this is bad practice, ADD experiments module to setup cfg
import sys
if "/Users/rcml/corel/" not in sys.path:
    sys.path.append("/Users/rcml/corel/")
from experiments.assets.data.rfp_fam import train_dataset
from experiments.assets.data.rfp_fam import test_dataset

from models import VAE


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


def _preprocess(x):
    tokenized_string = tf.strings.bytes_split(x)
    _x = aa_oh_lookup(tokenized_string)
    return _x, _x


if __name__ == "__main__":
    BATCHSIZE = 128
    EPOCHS = 500
    SEED = 42
    LR = 1e-3
    MODEL_PATH = "results/models/vae.ckpt"
    cpu = False
    if tf.test.gpu_device_name() != "/device:GPU:0":
        cpu = True
        warn("GPU device not found.")
    else:
        print(f"SUCCESS: Found GPU: {tf.test.gpu_device_name()}")

    # TODO: create TF dataset class from RFP MSA
    #datasets = tfds.load(name="rfp_fam", as_supervised=False)  FIXME: rfp_fam does not registered

    aa_oh_lookup = tf.keras.layers.StringLookup(vocabulary=AMINO_ACIDS, output_mode="one_hot")

    train_dataset = train_dataset.map(_preprocess).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE).shuffle(SEED)
    eval_dataset = test_dataset.map(_preprocess).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)

    #x0 = train_dataset.take(1).get_single_element()
    x0 = next(iter(train_dataset))[0][0]

    vae = VAE(z_dim=10, input_dims=x0.shape, n_categories=tf.constant(len(AMINO_ACIDS)))

    if cpu: # NOTE: M1/M2 processors require legacy Adam
        optimizer = tf.optimizers.legacy.Adam(learning_rate=LR)
    else:
        optimizer = tf.optimizers.Adam(learning_rate=LR)

    checkpoint_dir = os.path.dirname(MODEL_PATH)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=MODEL_PATH,
                                                    save_weights_only=True,
                                                    verbose=1)

    vae.model.compile(optimizer=optimizer, loss=lambda x, model: -model.log_prob(x))

    _ = vae.model.fit(train_dataset, epochs=EPOCHS, validation_data=eval_dataset, callbacks=[cp_callback])