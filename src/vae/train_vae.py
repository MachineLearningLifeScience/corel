import tensorflow as tf
from warnings import warn

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


if __name__ == "__main__":
    BATCHSIZE = 256
    EPOCHS = 15
    SEED = 42
    LR = 1e-3
    if tf.test.gpu_device_name() != "/device:GPU:0":
        warn("GPU device not found.")
    else:
        print(f"SUCCESS: Found GPU: {tf.test.gpu_device_name()}")

    # TODO: create TF dataset class from RFP MSA
    #datasets = tfds.load(name="rfp_fam", as_supervised=False)  FIXME: rfp_fam does not registered

    aa_oh_lookup = tf.keras.layers.StringLookup(vocabulary=AMINO_ACIDS, output_mode="one_hot")

    train_dataset = train_dataset.map(lambda x: tf.strings.bytes_split(x)).map(aa_oh_lookup).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE).shuffle(SEED)
    eval_dataset = test_dataset.map(lambda x: tf.strings.bytes_split(x)).map(aa_oh_lookup).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)

    #x0 = train_dataset.take(1).get_single_element()
    x0 = next(iter(train_dataset))

    vae = VAE(z_dim=10, input_dims=x0.shape, n_categories=tf.constant(len(AMINO_ACIDS)))
    optimizer = tf.optimizers.Adam(learning_rate=LR)
    vae.model.compile(optimizer=optimizer, loss=vae.neg_ll)

    _ = vae.model.fit(train_dataset, epochs=EPOCHS, validation_data=eval_dataset)