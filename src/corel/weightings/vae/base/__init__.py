## basic VAE architecture specifications
LATENT_DIM = 2
ENCODER_LAYERS = [1000, 250]
DECODER_LAYERS = [250, 1000]
KL_WEIGHT = 0.05
DROPOUT_RATE = 0.5
PRIOR_SCALE = 1

from .models import VAE
from .models import Encoder
from .models import Decoder
