import time
import sys

sys.path.append('..')

from utils import general_utils as utils
from model import id_encoder, latent_mapping, attr_encoder,\
    generator, discriminator, landmarks

from model.stylegan import StyleGAN_G, StyleGAN_D

import tensorflow as tf
from tensorflow.keras import layers, Model


class Network(Model):
    def __init__(self, args, id_net_path, base_generator,
                 landmarks_net_path=None, face_detection_model_path=None, test_id_net_path=None):
        super().__init__()
        self.args = args
        self.G = generator.G(args, id_net_path, base_generator,
                             landmarks_net_path, face_detection_model_path, test_id_net_path)

        if self.args.train:
            self.W_D = discriminator.W_D(args)

    def call(self):
        raise NotImplemented()

    def my_save(self, reason):
        self.G.my_save(reason)

        if self.args.W_D_loss:
            self.W_D.my_save(reason)

    def my_load(self):
        raise NotImplemented()

    def train(self):
        self._set_trainable_behavior(True)

    def test(self):
        self._set_trainable_behavior(False)

    def _set_trainable_behavior(self, trainable):
        self.G.attr_encoder.trainable = trainable
        self.G.latent_spaces_mapping.trainable = trainable
