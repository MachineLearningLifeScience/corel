## basic VAE architecture specifications
LATENT_DIM = 2
ENCODER_LAYERS = [1000, 100] # PREVIOUS 1000, 250
DECODER_LAYERS = [100, 1000] # PREVIOUS 250, 1000
KL_WEIGHT = 10 # 0.1 #float(200/188600) NOTE rule of thumb: batch_size / N_training \approx 128/188600 for BLAT
DROPOUT_RATE = 0.75 # last decoder layer
PRIOR_SCALE = 1
OFFDIAG = False #True # difference between MVN or Independent Normal prior

from .models import VAE
from .models import Encoder
from .models import Decoder
