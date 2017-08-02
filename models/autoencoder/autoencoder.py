import tensorflow as tf
from enum import Enum
from tensorflow.models.autoencoder.autoencoder_models.DenoisingAutoencoder import

class StackedAutoEncoder():

    def __init__(self, noise_type='masking'):
        self.noise_type = noise_type
        self.ae = None
        if noise_type = 'none':
            self.ae = Autoencoder()
        elif noise_type = 'gaussian':
            self.ae = GaussianNoiseAutoencoder()
        elif noise_type = 'masking':
            self.ae = MaskingNoiseAutoencoder()
        elif noise_type = 'variational':
            self.ae = VariationalAutoencoder()
        else:
            raise NameError

