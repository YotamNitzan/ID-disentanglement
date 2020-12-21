import logging

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input


class AttrEncoder(Model):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.logger = logging.getLogger(__class__.__name__)

        attr_encoder = InceptionV3(include_top=False, pooling='avg')
        self.model = attr_encoder

        if self.args.load_checkpoint:
            self.model.load_weights(str(self.args.load_checkpoint.joinpath(self.__class__.__name__ + '.h5')))

    @tf.function
    def call(self, input_x):
        x = tf.image.resize(input_x, (299, 299))
        x = preprocess_input(255 * x)
        x = self.model(x)
        x = tf.expand_dims(x, 1)

        return x

    def my_save(self, reason=''):
        self.model.save_weights(str(self.args.weights_dir.joinpath(self.__class__.__name__ + reason + '.h5')))
