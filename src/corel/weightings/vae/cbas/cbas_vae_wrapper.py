import os
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import default_float_type

from corel.weightings.vae.make_vae import build_vae


class CBASVAEWrapper:
    def __init__(self, AA: int, L: int, prefix):
        self.L = L
        self.AA = AA
        #prefix = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        vae_0 = build_vae(latent_dim=20,
                          n_tokens=self.AA, #20,  # TODO: I guess this is self.AA?
                          seq_length=self.L,
                          enc1_units=50)
        # unfortunately the call below is not usable
        #vae_0.load_all_weights()
        vae_suffix = "_5k_1"
        vae_0.encoder_.load_weights(prefix + "/vae_0_encoder_weights%s.h5" % vae_suffix)
        vae_0.decoder_.load_weights(prefix + "/vae_0_decoder_weights%s.h5"% vae_suffix)
        vae_0.vae_.load_weights(prefix + "/vae_0_vae_weights%s.h5"% vae_suffix)

        self.vae = vae_0

    def encode(self, x, grad=False):
        # seems like also for this VAE, the dimension of the amino acids is 1
        #x_ = tf.transpose(tf.reshape(x, [x.shape[0], self.L, self.AA]), [0, 2, 1])
        # TODO: set steps to 0?
        return self.vae.encoder_.predict(x)[0]

    def decode(self, z, grad=False):
        #return tf.cast(self.vae.decoder_.predict(z), default_float_type())
        return tf.cast(self.vae.decoder_(z), default_float_type())
