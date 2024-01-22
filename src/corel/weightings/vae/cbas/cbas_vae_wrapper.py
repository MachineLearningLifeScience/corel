from pathlib import Path
import numpy as np
import tensorflow as tf
# addition in tf.__version__ == 2.15.0
tf.experimental.numpy.experimental_enable_numpy_behavior(prefer_float32=False, dtype_conversion_mode="legacy")
from corel.weightings.vae.cbas.make_vae import build_vae


class CBASVAEWrapper:
    def __init__(self, AA: int, L: int, prefix):
        self.L = L
        self.AA = AA
        vae_0 = build_vae(latent_dim=20,
                          n_tokens=self.AA, #20,  # TODO: I guess this is self.AA?
                          seq_length=self.L,
                          enc1_units=50)
        # unfortunately the call below is not usable
        vae_suffix = "_5k_1"
        vae_0.encoder_.load_weights(prefix + "/vae_0_encoder_weights%s.h5" % vae_suffix)
        vae_0.decoder_.load_weights(prefix + "/vae_0_decoder_weights%s.h5"% vae_suffix)
        vae_0.vae_.load_weights(prefix + "/vae_0_vae_weights%s.h5"% vae_suffix)

        self.vae = vae_0
        self.training_data = Path(prefix).parent.parent.resolve() / "gfp_data.csv"

    def encode(self, x, grad=False):
        # seems like also for this VAE, the dimension of the amino acids is 1
        #x_ = tf.transpose(tf.reshape(x, [x.shape[0], self.L, self.AA]), [0, 2, 1])
        # TODO: set steps to 0?
        return self.vae.encoder_.predict(x)[0]

    def decode(self, z, grad=False):
        #return tf.cast(self.vae.decoder_.predict(z), default_float_type())
        #return tf.cast(self.vae.decoder_(z), default_float_type())
        return tf.cast(self.vae.decoder_(z), tf.float64)

    def get_training_data(self) -> np.ndarray:
        # TODO: filter to constant length, do NOT return all available data, only the one used during training!
        if not self.training_data.exists():
            raise ValueError("The specified path does not contain GFP training data!")
        train_data = np.genfromtxt(str(self.training_data), delimiter=",", skip_header=1, dtype=None)
        train_sequences = [x[-1].decode("utf-8") for x in train_data]
        train_sequences_unique = []
        for s in train_sequences:  # obtain unique sequences
            if s not in train_sequences_unique:
                train_sequences_unique.append(s)
        train_sequences_unique = [s for s in train_sequences_unique if len(s) == self.L] # filter to aligned length of unlabelled sequences
        return train_sequences_unique
