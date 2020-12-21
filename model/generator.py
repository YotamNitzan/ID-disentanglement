import logging

from model import id_encoder
from model import attr_encoder
from model import latent_mapping
from model import landmarks
from model.arcface.inference import MyArcFace

import tensorflow as tf
from tensorflow.keras import layers, Model


class G(Model):
    def __init__(self, args, id_model_path, image_G,
                 landmarks_net_path, face_detection_model_path, test_id_model_path):

        super().__init__()
        self.args = args
        self.logger = logging.getLogger(__class__.__name__)

        self.id_encoder = id_encoder.IDEncoder(args, id_model_path)
        self.id_encoder.trainable = False

        self.attr_encoder = attr_encoder.AttrEncoder(args)

        self.latent_spaces_mapping = latent_mapping.LatentMappingNetwork(args)

        self.stylegan_s = image_G
        self.stylegan_s.trainable = False

        if args.train:
            self.test_id_encoder = MyArcFace(test_id_model_path)
            self.test_id_encoder.trainable = False

            self.landmarks = landmarks.LandmarksDetector(args, landmarks_net_path, face_detection_model_path)
            self.landmarks.trainable = False

    @tf.function
    def call(self, x1, x2):
        id_embedding = self.id_encoder(x1)

        lnds = self.landmarks(x2)
        attr_input = x2

        attr_out = self.attr_encoder(attr_input)
        attr_embedding = attr_out
        z_tag = tf.concat([id_embedding, attr_embedding], -1)
        w = self.latent_spaces_mapping(z_tag)

        out = self.stylegan_s(w)

        # Move to roughly [0,1]
        out = (out + 1) / 2

        return out, id_embedding,  attr_out, w[:, 0, :], lnds

    def my_save(self, reason=''):
        self.attr_encoder.my_save(reason)
        self.latent_spaces_mapping.my_save(reason)


