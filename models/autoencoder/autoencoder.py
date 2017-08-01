import tensorflow as tf
from tensorflow.models.autoencoder.autoencoder_models.DenoisingAutoencoder import MaskingNoiseAutoencoder

class AutoEncoderFactory:

    def __init__(self, BaseAutoEncoder):
        self.ae = 