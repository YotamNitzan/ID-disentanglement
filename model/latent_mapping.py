from utils.general_utils import get_weights

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model


class LatentMappingNetwork(Model):
    def __init__(self, args):
        super().__init__()
        self.args = args

        input_shape = (2560,)

        self.linear1 = layers.Dense(2048, input_shape=input_shape)
        self.linear2 = layers.Dense(1024)
        self.linear3 = layers.Dense(512, kernel_initializer=get_weights())
        self.linear4 = layers.Dense(512, kernel_initializer=get_weights())
        self.linears = [self.linear1, self.linear2, self.linear3, self.linear4]

        self.relu = layers.LeakyReLU(0.2)

        self.num_styles = int(np.log2(self.args.resolution)) * 2 - 2

        if self.args.load_checkpoint:
            self.build(input_shape=(1, 1, 2560))
            self.load_weights(str(self.args.load_checkpoint.joinpath(self.__class__.__name__ + '.h5')))

    @tf.function
    def call(self, x):
        first = True
        for layer in self.linears:
            if not first:
                x = self.relu(x)

            x = layer(x)
            first = False

        s = list(x.shape)

        # Duplicate the column vector w along columns for each AdaIN entry
        s[1] = self.num_styles
        x = tf.broadcast_to(x, s)

        return x

    def my_save(self, reason=''):
        self.save_weights(str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + '.h5')))
