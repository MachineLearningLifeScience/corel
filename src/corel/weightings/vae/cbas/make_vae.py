"""
CBAS VAE architecture and training from existing models.
"""

import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import (Add, Dense, Flatten, Input, Lambda, Layer,
                                     Multiply, Reshape)
from tensorflow.keras.models import Model
from tensorflow.python.keras import backend as K

import corel
from corel.weightings.vae.cbas.losses import (identity_loss,
                                              summed_categorical_crossentropy)
from corel.weightings.vae.cbas.util import get_experimental_X_y

# added by Simon
MAKE_DETERMINISTIC = True


def train_experimental_vaes(latent_dim=20):
    """Trains and saves VAEs on the GFP data for use in the weighted ML methods"""
    TRAIN_SIZE = 5000
    train_size_str = f"{(TRAIN_SIZE/1000)}k" 
    suffix = f"_{train_size_str}"
    if latent_dim != 20:
        suffix += f"_d{latent_dim}"
    for i in [0, 2]:
        RANDOM_STATE = i + 1
        X_train, _, _  = get_experimental_X_y(random_state=RANDOM_STATE, train_size=TRAIN_SIZE)
        vae_0 = build_vae(latent_dim=latent_dim,
                  n_tokens=20,
                  seq_length=X_train.shape[1],
                  enc1_units=50)
        vae_0.fit([X_train], 
                  [X_train, np.zeros(X_train.shape[0])],
                  epochs=100,
                  batch_size=10,
                  verbose=2)
        output_path = Path(corel.__file__).parent.parent.resolve() / "results" / "models"
        vae_0.encoder_.save_weights(str(output_path) + f"/models/vae_0_encoder_weights{suffix}_{RANDOM_STATE}.h5")
        vae_0.decoder_.save_weights(str(output_path) + f"/models/vae_0_decoder_weights{suffix}_{RANDOM_STATE}.h5")
        vae_0.vae_.save_weights(str(output_path) + f"/models/vae_0_vae_weights{suffix}_{RANDOM_STATE}.h5")


def build_vae(latent_dim, n_tokens=4, seq_length=33, enc1_units=50, eps_std=1., ):
    """Returns a compiled VAE model"""
    model = SimpleVAE(input_shape=(seq_length, n_tokens,),
                      latent_dim=latent_dim)

    # set encoder layers:
    model.encoderLayers_ = [
        Dense(units=enc1_units, activation='elu', name='e2'),
    ]

    # set decoder layers:
    model.decoderLayers_ = [
        Dense(units=enc1_units, activation='elu', name='d1'),
        Dense(units=n_tokens * seq_length, name='d3'),
        Reshape((seq_length, n_tokens), name='d4'),
        Dense(units=n_tokens, activation='softmax', name='d5'),
    ]

    # build models:
    kl_scale = K.variable(1.)
    model.build_encoder()
    model.build_decoder(decode_activation='softmax')
    model.build_vae(epsilon_std=eps_std, kl_scale=kl_scale)

    losses = [summed_categorical_crossentropy, identity_loss]

    model.compile(optimizer='adam',
                  loss=losses)

    return model

"""
Module for extendable variational autoencoders.

Some code adapted from Louis Tiao's blog: http://louistiao.me/
"""


class KLDivergenceLayer(Layer):
    """ Add KL divergence in latent layer to loss """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs, scale=1.):
        """ Add KL loss, then return inputs """

        mu, log_var = inputs
        inner = 1 + log_var - K.square(mu) - K.exp(log_var)

        # sum over dimensions of latent space
        kl_batch = -0.5 * K.sum(inner, axis=1)

        # add mean KL loss over batch
        self.add_loss(scale * K.mean(kl_batch, axis=0), inputs=inputs)
        return mu, log_var


class KLScaleUpdate(Callback):
    """ Callback for updating the scale of the the KL divergence loss

    See Bowman et. al (2016) for motivation on adjusting the scale of the
    KL loss. This class implements a sigmoidal growth, as in Bowman, et. al.

    """

    def __init__(self, scale, growth=0.01, start=0.001, verbose=True):
        super(KLScaleUpdate, self).__init__()
        self.scale_ = scale
        self.start_ = start
        self.growth_ = growth
        self.step_ = 0
        self.verbose_ = verbose

    def on_batch_begin(self, batch, logs=None):
        K.set_value(self.scale_, self._get_next_val(self.step_))
        self.step_ += 1

    def _get_next_val(self, step):
        return 1 - (1 / (1 + self.start_ * np.exp(step * self.growth_)))

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose_ > 0:
            print("KL Divergence weight: %.3f" % K.get_value(self.scale_))


class BaseVAE(object):
    """ Base class for Variational Autoencoders implemented in Keras

    The class is designed to connect user-specified encoder and decoder
    models via a Model representing the latent space

    """

    def __init__(self, input_shape, latent_dim, *args, **kwargs):
        self.latentDim_ = latent_dim
        self.inputShape_ = input_shape

        self.encoder_ = None
        self.decoder_ = None

        self.vae_ = None

    def build_encoder(self, *args, **kwargs):
        """ Build the encoder network as a keras Model

        The encoder Model must ouput the mean and log variance of
        the latent space embeddings. I.e. this model must output
        mu and Sigma of the latent space distribution:

                    q(z|x) = N(z| mu(x), Sigma(x))

        Sets the value of self.encoder_ to a keras Model

        """

        raise NotImplementedError

    def build_decoder(self, *args, **kwargs):
        """ Build the decoder network as a keras Model

        The input to the decoder must have the same shape as the latent
        space and the output must have the same shape as the input to
        the encoder.

        Sets the value of self.decoder_ to a keras Model

        """

        raise NotImplementedError

    def _build_latent_vars(self, mu_z, log_var_z, epsilon_std=1., kl_scale=1.):
        """ Build keras variables representing the latent space

        First, calculate the KL divergence from the input mean and log variance
        and add this to the model loss via a KLDivergenceLayer. Then sample an epsilon
        and perform a location-scale transformation to obtain the latent embedding, z.

        Args:
            epsilon_std: standard deviation of p(epsilon)
            kl_scale: weight of KL divergence loss

        Returns:
            Variables representing z and epsilon

        """

        # mu_z, log_var_z, kl_batch  = KLDivergenceLayer()([mu_z, log_var_z], scale=kl_scale)
        lmda_func = lambda inputs: -0.5 * K.sum(1 + inputs[1] - K.square(inputs[0]) - K.exp(inputs[1]), axis=1)

        kl_batch = Lambda(lmda_func, name='kl_calc')([mu_z, log_var_z])
        kl_batch = Reshape((1,), name='kl_reshape')(kl_batch)

        # get standard deviation from log variance:
        sigma_z = Lambda(lambda lv: K.exp(0.5 * lv))(log_var_z)

        if MAKE_DETERMINISTIC:
            eps = Input(tensor=K.zeros_like(mu_z))
        else:
            # re-parametrization trick ( z = mu_z + eps * sigma_z)
            eps = Input(tensor=K.random_normal(stddev=epsilon_std,
                                               shape=(K.shape(mu_z)[0], self.latentDim_)))

        eps_z = Multiply()([sigma_z, eps])  # scale by epsilon sample
        z = Add()([mu_z, eps_z])

        return z, eps, kl_batch

    def _get_decoder_input(self, z, enc_in):
        return z

    def build_vae(self, epsilon_std=1., kl_scale=1.):
        """ Build the VAE

        Sets the value of self.vae_ to a keras model

        Args:
            epsilon_std (float): standard deviation of distribution used to sample z via the reparameratization trick
            kl_scale (float or keras.backend.Variable): weight of KL divergence loss

        """

        if self.encoder_ is None:
            raise TypeError("Encoder must be built before calling build_vae")
        if self.decoder_ is None:
            raise TypeError("Decoder must be built before calling build_vae")

        enc_in = self.encoder_.inputs
        mu_z, log_var_z = self.encoder_.outputs
        z, eps, kl_batch = self._build_latent_vars(mu_z, log_var_z, epsilon_std=epsilon_std, kl_scale=kl_scale)
        dec_in = self._get_decoder_input(z, enc_in)
        x_pred = self.decoder_(dec_in)
        self.vae_ = Model(inputs=enc_in + [eps], outputs=[x_pred, kl_batch], name='vae_base')

    def plot_model(self, *args, **kwargs):
        keras.utils.plot_model(self.vae_, *args, **kwargs)

    def compile(self, *args, **kwargs):
        self.vae_.compile(*args, **kwargs)

    def fit(self, *args, **kwargs):
        self.vae_.fit(*args, **kwargs)

    def save_all_weights(self, prefix):
        encoder_file = prefix + "_encoder.h5"
        decoder_file = prefix + "_decoder.h5"
        vae_file = prefix + "_vae.h5"

        self.encoder_.save_weights(encoder_file)
        self.decoder_.save_weights(decoder_file)
        self.vae_.save_weights(vae_file)

    def load_all_weights(self, prefix):
        encoder_file = prefix + "_encoder.h5"
        decoder_file = prefix + "_decoder.h5"
        vae_file = prefix + "_vae.h5"

        self.encoder_.load_weights(encoder_file)
        self.decoder_.load_weights(decoder_file)
        self.vae_.load_weights(vae_file)


class BaseSupervisedVAE(BaseVAE):
    """ Base class for VAEs that also make predictions from the latent space """

    def __init__(self, input_shape, latent_dim, pred_dim,
                 learn_uncertainty=False, pred_var=0.1, *args, **kwargs):
        super(BaseSupervisedVAE, self).__init__(input_shape=input_shape,
                                                latent_dim=latent_dim,
                                                *args, **kwargs)
        self.predDim_ = pred_dim
        self.predictor_ = None
        self.learnUncertainty_ = learn_uncertainty
        self.predVar_ = pred_var

    def build_encoder(self, *args, **kwargs):
        raise NotImplementedError

    def build_decoder(self, *args, **kwargs):
        raise NotImplementedError

    def build_predictor(self, *args, **kwargs):
        """ Build the predictor network as a keras Model

        The input to the predictor must have the same shape as the latent
        space and the output must have self.predShape_

        Sets the value of self.predictor_ to a keras Model

        """
        raise NotImplementedError

    def build_vae(self, epsilon_std=1., kl_scale=1.):
        """ Build the VAE

        Sets the value of self.vae_ to a keras model

        Args:
            epsilon_std (float): standard deviation of distribution used to sample z via the reparameratization trick
            kl_scale (float or keras.backend.Variable): weight of KL divergence loss

        """

        if self.encoder_ is None:
            raise TypeError("Encoder must be built before calling build_vae")
        if self.decoder_ is None:
            raise TypeError("Decoder must be built before calling build_vae")

        enc_in = self.encoder_.inputs
        mu_z, log_var_z = self.encoder_.outputs
        z, eps, kl_batch = self._build_latent_vars(mu_z, log_var_z, epsilon_std=epsilon_std, kl_scale=kl_scale)
        dec_in = self._get_decoder_input(z, enc_in)
        x_pred = self.decoder_(dec_in)
        y_pred, y_log_var = self.predictor_(dec_in)
        self.vae_ = Model(inputs=enc_in + [eps], outputs=[x_pred, kl_batch, y_pred, y_log_var], name='vae_pred')

    def save_all_weights(self, prefix):
        super(BaseSupervisedVAE, self).save_all_weights(prefix)
        predictor_file = prefix + "_predictor.h5"
        self.predictor_.save_weights(predictor_file)

    def load_all_weights(self, prefix):
        super(BaseSupervisedVAE, self).load_all_weights(prefix)
        predictor_file = prefix + "_predictor.h5"
        self.predictor_.load_weights(predictor_file)


class SimpleVAE(BaseVAE):
    """ Basic VAE where the encoder and decoder can be constructed from lists of layers """

    def __init__(self, input_shape, latent_dim, flatten=True, *args, **kwargs):
        super(SimpleVAE, self).__init__(input_shape=input_shape,
                                        latent_dim=latent_dim,
                                        *args, **kwargs)
        self.flatten_ = flatten
        self.encoderLayers_ = []
        self.decoderLayers_ = []

    def add_encoder_layer(self, layer):
        """ Append a keras Layer to self.encoderLayers_"""
        self.encoderLayers_.append(layer)

    def add_decoder_layer(self, layer):
        """ Append a keras Layer to self.decoderLayers_ """
        self.decoderLayers_.append(layer)

    def _build_encoder_inputs(self):
        """ BUILD (as opposed to get) the encoder inputs """
        x = Input(shape=self.inputShape_)
        return [x]

    def _build_decoder_inputs(self):
        z = Input(shape=(self.latentDim_,))
        return z

    def _edit_encoder_inputs(self, enc_in):
        if self.flatten_:
            h = Flatten()(enc_in[0])
        else:
            h = enc_in[0]
        return h

    def _edit_decoder_inputs(self, dec_in):
        return dec_in

    def build_encoder(self):
        """ Construct the encoder from list of layers

        After the final layer in self.encoderLayers_, two Dense layers
        are applied to output mu_z and log_var_z

        """

        if len(self.encoderLayers_) == 0:
            raise ValueError("Must add at least one encoder hidden layer")

        enc_in = self._build_encoder_inputs()
        h = self._edit_encoder_inputs(enc_in)
        for hid in self.encoderLayers_:
            h = hid(h)

        mu_z = Dense(self.latentDim_, name='mu_z')(h)
        log_var_z = Dense(self.latentDim_, name='log_var_z')(h)

        self.encoder_ = Model(inputs=enc_in, outputs=[mu_z, log_var_z], name='encoder')

    def build_decoder(self, decode_activation):
        """ Construct the decoder from list of layers

        After the final layer in self.decoderLayers_, a Dense layer is
        applied to output the final reconstruction

        Args:
            decode_activation: activation of the final decoding layer

        """

        if len(self.decoderLayers_) == 0:
            raise ValueError("Must add at least one decoder hidden layer")

        dec_in = self._build_decoder_inputs()
        h = self._edit_decoder_inputs(dec_in)
        for hid in self.decoderLayers_:
            h = hid(h)

        x_pred = h
        self.decoder_ = Model(inputs=dec_in, outputs=x_pred, name='decoder')


class SimpleSupervisedVAE(SimpleVAE, BaseSupervisedVAE):
    """ Supervised VAE where the predictor, encoder and decoder can be built from a list of layers """

    def __init__(self, input_shape, latent_dim, pred_dim, pred_var=0.1, learn_uncertainty=False):
        super(SimpleSupervisedVAE, self).__init__(input_shape=input_shape,
                                                  latent_dim=latent_dim,
                                                  pred_var=pred_var,
                                                  pred_dim=pred_dim,
                                                  learn_uncertainty=learn_uncertainty)
        self.predictorLayers_ = []

    def add_predictor_layer(self, layer):
        """ Append a keras Layer to self.predictorLayers_ """
        self.predictorLayers_.append(layer)

    def build_predictor(self, predict_activation=None):
        """ Construct the predictor network from the list of layers

        After the last layer in self.predictorLayers_, a final Dense layer is added
        that with self.predDim_ units (i.e. outputs the prediction)

        Args:
            predict_activation: activation function for the final dense layer

        """

        if len(self.predictorLayers_) == 0:
            raise ValueError("Must add at least one predictor hidden layer")

        pred_in = self._build_decoder_inputs()
        h = self._edit_decoder_inputs(pred_in)
        for hid in self.predictorLayers_:
            h = hid(h)

        y_pred = Dense(units=self.predDim_,
                       activation=predict_activation)(h)
        log_var_y = Dense(self.predDim_, name='log_var_y')(h)

        if not self.learnUncertainty_:
            log_var_y = Lambda(lambda lv: 0 * lv + K.ones_like(lv) * K.log(K.variable(self.predVar_)))(log_var_y)

        self.predictor_ = Model(inputs=pred_in, outputs=[y_pred, log_var_y], name='predictor')

    def build_vae(self, epsilon_std=1., kl_scale=1.):
        BaseSupervisedVAE.build_vae(self, epsilon_std=epsilon_std, kl_scale=kl_scale)

    def save_all_weights(self, prefix):
        BaseSupervisedVAE.save_all_weights(self, prefix)

    def load_all_weughts(self, prefix):
        BaseSupervisedVAE.load_all_weights(self, prefix)


if __name__ =='__main__':
    # train_experimental_vaes()
    train_experimental_vaes(latent_dim=2)
