import numpy as np
from base.autoencoder_models.Autoencoder import Autoencoder
from base.autoencoder_models.VariationalAutoencoder import VariationalAutoencoder
from base.autoencoder_models.DenoisingAutoencoder import MaskingNoiseAutoencoder
from base.autoencoder_models.DenoisingAutoencoder import AdditiveGaussianNoiseAutoencoder as GaussianNoiseAutoencoder


class StackedAutoEncoder:

    def __init__(self, layers=(40, 20, 10, 5), epochs=(300, 200, 100), batch_size = 8, noise_type='masking'):
        assert(len(layers) > 1)
        assert(len(epochs) == len(layers) - 1)
        self.noise_type = noise_type
        self.ae = []
        self.layers = layers
        self.epochs = epochs
        self.batch_size = batch_size

        for i in range(len(layers)-1):
            if noise_type is None:
                self.ae.append(Autoencoder(layers[i], layers[i+1]))
            elif noise_type == 'gaussian':
                self.ae.append(GaussianNoiseAutoencoder(layers[i], layers[i+1]))
            elif noise_type == 'masking':
                self.ae.append(MaskingNoiseAutoencoder(layers[i], layers[i+1]))
            elif noise_type == 'variational':
                self.ae.append(VariationalAutoencoder(layers[i], layers[i+1]))
            else:
                raise ValueError

    def train(self, X):
        inputs = X
        for i in range(1, len(self.layers)):
            inputs = self._train_layer(inputs, i)

    def _train_layer(self, inputs, hidden_layer_ndx):
        for epoch in range(self.epochs[hidden_layer_ndx]):
            avg_cost = 0.
            total_batch = int(len(inputs) / self.batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = self._get_random_block_from_data(inputs, self.batch_size)

                # Fit training using batch data
                cost = self.ae[hidden_layer_ndx].partial_fit(batch_xs)
                # Compute average loss
                avg_cost += cost / len(inputs) * self.batch_size

            # Display logs per epoch step
            if epoch % 10 == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

        print("Total cost: " + str(self.ae[hidden_layer_ndx].calc_total_cost(inputs)))

        return self.ae[hidden_layer_ndx].hidden

    def _get_random_block_from_data(self, data, batch_size):
        start_index = np.random.randint(0, len(data) - batch_size)
        return data[start_index:(start_index + batch_size)]
