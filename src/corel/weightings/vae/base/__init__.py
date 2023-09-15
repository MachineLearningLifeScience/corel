## basic VAE architecture specifications
LATENT_DIM = 2
ENCODER_LAYERS = [1500, 1500] # PREVIOUS 1000, 250
DECODER_LAYERS = [100, 2000] # PREVIOUS 250, 1000
KL_WEIGHT = 0.01 # PREVIOUS: 0.025
DROPOUT_RATE = 0.5
PRIOR_SCALE = 1

from .models import VAE
from .models import Encoder
from .models import Decoder
