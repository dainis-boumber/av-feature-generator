import tensorflow
import base.autoencoder_models.AdditiveGaussianNoiseAutoencoder as GaussianNoiseAutoencoder
from base.autoencoder_models import Autoencoder, VariationalAutoencoder
from base.autoencoder_models.DenoisingAutoencoder import MaskingNoiseAutoencoder


class StackedAutoEncoder():

    def __init__(self, noise_type='masking'):
        self.noise_type = noise_type
        self.ae = None
        if noise_type is None:
            self.ae = Autoencoder()
        elif noise_type == 'gaussian':
            self.ae = GaussianNoiseAutoencoder()
        elif noise_type == 'masking':
            self.ae = MaskingNoiseAutoencoder()
        elif noise_type == 'variational':
            self.ae = VariationalAutoencoder()
        else:
            raise ValueError

