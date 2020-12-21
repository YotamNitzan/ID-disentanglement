import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

from utils import general_utils as utils
from model.face_detector import FaceDetector


class LandmarksDetector(Model):
    def __init__(self, args, model_path, face_detection_model_path):
        super().__init__()
        self.args = args
        self.face_detector = FaceDetector(args, face_detection_model_path)
        self.expand_ratio = 0.2

        # Load without source code
        self.model = tf.saved_model.load(model_path)

    # Preprocess
    def preprocess(self, imgs, face_detection=False):
        imgs *= 255
        if face_detection:
            imgs, details = self.hard_preprocess(imgs)
        else:
            imgs, details = self.lazy_preprocess(imgs)

        return imgs, details

    def lazy_preprocess(self, imgs):
        imgs = tf.image.resize(imgs, (160, 160))
        return imgs, 160

    def hard_preprocess(self, imgs):
        bboxes = self.face_detector(imgs)

        centers = np.array([bboxes[:, 0] + bboxes[:, 2], bboxes[:, 1] + bboxes[:, 3]]).T // 2

        # Duplicate center point into column order of x,x,y,y
        centers = np.repeat(centers, repeats=2, axis=1)

        # Permute columns order into x,y,x,y
        centers[:] = utils.np_permute(centers, [0, 2, 1, 3])

        # Calculate widths of current bboxes
        widths = np.transpose([bboxes[:, 2] - bboxes[:, 0]])

        # Calculate the maximal expansion
        max_expand = int(np.ceil(np.max(widths) * self.expand_ratio))

        # Pad the image with the maximal expansion.
        # Useful in case an expanded bounding box goes outside image
        paddings = tf.constant([[0, 0], [max_expand, max_expand], [max_expand, max_expand], [0, 0]])
        pad_imgs = tf.pad(imgs, paddings, mode='CONSTANT', constant_values=127.)

        # The size of the new square bounding box
        new_scales = np.floor((1 + 2 * self.expand_ratio) * widths)

        # Size of step from the center
        new_half_scales = new_scales // 2

        # Repeat step in all directions
        # Decrease in start point, Increase in end point
        new_half_scales = np.repeat(new_half_scales, repeats=4, axis=1) * [-1, -1, 1, 1]

        # Bounding boxes in respect to padded image
        new_bboxes = centers + new_half_scales + max_expand

        # tf.image.crop_and_resize requires bounding boxes to be normalized
        # i.e., between [0,1] and also in order (y,x)
        normed_bboxes = utils.np_permute(new_bboxes, [1, 0, 3, 2]) / pad_imgs.shape[1]

        cropped_imgs = tf.image.crop_and_resize(pad_imgs, normed_bboxes,
                                                box_indices=range(self.args.batch_size), crop_size=(160, 160))

        details = (new_scales, new_bboxes[:,:2], max_expand)
        return cropped_imgs, details

    # Postprocess
    def postprocess(self, landmarks, details, face_detection=False):
        landmarks = tf.reshape(landmarks, [-1, 68, 2])

        if face_detection:
            return self.hard_postprocess(landmarks, details)
        else:
            return self.lazy_postprocess(landmarks, details)

    def lazy_postprocess(self, batch_lnds, details):
        scale = details
        return scale * batch_lnds

    def hard_postprocess(self, batch_lnds, details):
        scale, from_origin, pad = details

        scale = tf.broadcast_to(scale, [scale.shape[0], 2])
        scale = tf.expand_dims(scale, axis=1)

        from_origin = tf.expand_dims(from_origin, axis=1)
        from_origin = tf.cast(from_origin, tf.dtypes.float32)

        lnds = batch_lnds * scale + from_origin - pad
        return lnds

    @tf.function
    def call(self, input_x, face_detection=False):

        # The network input format is a uint8 image (0-255) but in float32 dtype. ^__('')__^
        x, details = self.preprocess(input_x, face_detection)

        batch_lnds = self.model.inference(x)['landmark']

        batch_lnds = self.postprocess(batch_lnds, details, face_detection)

        return batch_lnds[:, 17:, :]
