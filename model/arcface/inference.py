import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import cv2
from model.arcface.resnet import ResNet50, train_model
from mtcnn import MTCNN
from skimage import transform as trans


class MyArcFace:
    def __init__(self, path_to_weights):
        self.model = train_model()
        self.model.load_weights(path_to_weights)
        self.model_resnet = self.model.resnet
        self.model.resnet.trainable = False
        self.mtcnn = MTCNN(min_face_size=80)

    def get_best_face(self, faces, resolution):
        if len(faces) == 0:
            raise IndexError('No faces found')
        if len(faces) == 1:
            return faces[0]

        print('Found more than one face')

        indices = list(range(len(faces)))

        # filter low confidence
        new_indices = [ind for ind in indices if faces[ind]['confidence'] > 0.99]
        # print(f'after confidence filtering: {len(new_indices)}')
        if len(new_indices) == 1:
            return faces[new_indices[0]]
        elif len(new_indices) > 1:
            indices = new_indices

        # filter not centered, distance between x and y must relatively small
        new_indices = [ind for ind in indices if np.abs(faces[ind]['box'][0] - faces[ind]['box'][1]) < resolution / 2.5]
        # print(f'after center filtering: {len(new_indices)}')
        if len(new_indices) == 1:
            return faces[new_indices[0]]
        elif len(new_indices) > 1:
            indices = new_indices

        # Take box with biggest height
        ind = max(indices, key=lambda ind: faces[ind]['box'][-1])
        return faces[ind]

    def __detect_face(self, img):
        # The assumption is that the image is RGB
        faces = self.mtcnn.detect_faces(img)
        face_obj = self.get_best_face(faces, img.shape[0])

        face_box_obj = face_obj['box']
        face_landmarks_obj = face_obj['keypoints']
        face_landmarks = np.zeros((5, 2))
        face_landmarks[0] = [face_landmarks_obj['left_eye'][0], face_landmarks_obj['right_eye'][1]]
        face_landmarks[1] = [face_landmarks_obj['right_eye'][0], face_landmarks_obj['left_eye'][1]]
        face_landmarks[2] = [face_landmarks_obj['nose'][0], face_landmarks_obj['nose'][1]]
        face_landmarks[3] = [face_landmarks_obj['mouth_left'][0], face_landmarks_obj['mouth_right'][1]]
        face_landmarks[4] = [face_landmarks_obj['mouth_right'][0], face_landmarks_obj['mouth_left'][1]]
        x = face_box_obj[0]
        y = face_box_obj[1]
        w = face_box_obj[2]
        h = face_box_obj[3]
        face_box = [x, y, x + w, y + h]
        return face_box, face_landmarks

    def __preprocess(self, img, bbox=None, landmark=None):
        M = None
        image_size = [112, 112]
        assert landmark is not None
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)
        tform = trans.SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params
        assert M is not None
        transforms = np.array(M).flatten()[:-1]
        tf_transforms = tf.constant([transforms], tf.float32)
        img_tensor = tf.convert_to_tensor(img.astype(np.float32))
        batch = tf.stack([img_tensor])
        output = tfa.image.transform(batch, tf_transforms, interpolation='BILINEAR', output_shape=image_size)
        return output

    def process_image(self, img):

        if (isinstance(img, tf.Tensor) and img.dtype != tf.dtypes.uint8) or img.dtype != np.uint8:
            img = np.uint8(img * 255)

        face_box, face_landmarks = self.__detect_face(img)
        aligned_face = self.__preprocess(img, face_box, face_landmarks)
        aligned_face -= 127.5
        aligned_face *= 0.0078125
        embeddings = self.model_resnet(aligned_face)
        normelized_embeddings = tf.math.l2_normalize(embeddings)
        return normelized_embeddings

    def __call__(self, img):
        if img.ndim == 4:
            embedding_list = []
            for x in img:
                norm_embedding = self.process_image(x)
                embedding_list.append(norm_embedding)
            return np.array(embedding_list)
        else:
            return self.process_image(img)
