import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class IDEncoder(Model):

    def __init__(self, args, model_path, intermediate_layers_names=None):
        super().__init__()
        self.args = args
        self.mean = (91.4953, 103.8827, 131.0912)
        base_model = tf.keras.models.load_model(model_path)

        if intermediate_layers_names:
            outputs = [base_model.get_layer(name).output for name in intermediate_layers_names]
        else:
            outputs = []

        # Add output of the network in any case
        outputs.append(base_model.layers[-2].output)

        self.model = tf.keras.Model(base_model.inputs, outputs)


    def crop_faces(self, img):
        ps = []
        for i in range(img.shape[0]):
            oneimg = img[i]
            try:
                box = tf.numpy_function(self.mtcnn.detect_faces, [oneimg], np.uint8)
                box = [z.numpy() for z in box[:4]]

                x1, y1, w, h = box

                x_expand = w * 0.3
                y_expand = h * 0.3

                x1 = int(np.maximum(x1 - x_expand // 2, 0))
                y1 = int(np.maximum(y1 - y_expand // 2, 0))

                x2 = int(np.minimum(x1 + w + x_expand // 2, self.args.resolution))
                y2 = int(np.minimum(y1 + h + y_expand // 2, self.args.resolution))
            except Exception as e:
                x1, y1, x2, y2 = 24, 50, 224, 250

            p = oneimg[y1:y2, x1:x2, :]
            p = tf.convert_to_tensor(p)
            p = tf.image.resize(p, (self.args.resolution, self.args.resolution))
            ps.append(p)

        ps = tf.stack(ps, 0)
        return ps

    def preprocess(self, img):
        """
        In VGGFace2 The preprocessing is:
            1. Face detection
            2. Expand bbox by factor of 0.3
            3. Resize so shorter side is 256
            4. Crop center 224x224

        In StyleGAN faces are not in-the-wild, we get an image of the head.
        Just cropping a loose center instead of face detection
        """

        # Go from [0, 1] to [0, 255]
        img = 255 * img

        min_x = int(0.1 * self.args.resolution)
        max_x = int(0.9 * self.args.resolution)
        min_y = int(0.1 * self.args.resolution)
        max_y = int(0.9 * self.args.resolution)

        img = img[:, min_x:max_x, min_y:max_y, :]
        img = tf.image.resize(img, (256, 256))

        start = (256 - 224) // 2
        img = img[:, start: 224 + start, start: 224 + start, :]
        img = img[:, :, :, ::-1] - self.mean

        return img

    @tf.function
    def call(self, input_x, get_intermediate=False):
        x = self.preprocess(input_x)
        x = self.model(x)

        if isinstance(x, list):
            embedding = x[-1]
            intermediates = x[:-1]
        else:
            embedding = x
            intermediates = None

        embedding = tf.math.l2_normalize(embedding, axis=-1)
        embedding = tf.expand_dims(embedding, 1)

        if get_intermediate and intermediates:
            return embedding, intermediates
        else:
            return embedding
