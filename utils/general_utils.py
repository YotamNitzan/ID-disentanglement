from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
import numbers
import scipy
import dlib

landmarks_model_path = None


def read_image(img_path, resolution, align=False):
    if align:
        img = read_and_align_image(img_path, resolution)
    else:
        img = read_SG_image(img_path, resolution)

    return img


def find_file_by_str(search_dir, s):
    files = [f for f in search_dir.iterdir() if s in f.name]
    return files


def read_SG_image(img_path, size=256, resize=True):
    img = Image.open(str(img_path))
    img = img.convert('RGB')

    if img.size != (size, size) and resize:
        img = img.resize((size, size))
    img = np.asarray(img)

    img = np.expand_dims(img, axis=0)

    # Images in [0, 1]
    img = np.float32(img) / 255

    return img


def read_and_align_image(img_path, output_size=1024):
    global landmarks_model_path
    if not landmarks_model_path:
        raise ValueError('Please init the landmarks model path')

    transform_size = 4096
    enable_padding = True

    img = Image.open(img_path)
    img = img.convert('RGB')
    npimg = np.asarray(img)

    # states is a 4x1 array with confidence for : [left eye closed, right eye closed, mouth closed, mouth open big]
    face_detector = dlib.get_frontal_face_detector()
    landmarks_network = dlib.shape_predictor(landmarks_model_path)

    try:
        bbox = face_detector(npimg, 0)[0]
    except:
        print('face not found!')
        raise

    # rect = np.array([det.left(), det.top(), det.right(), det.bottom()])
    shape = landmarks_network(npimg, bbox)
    lm = np.array([[shape.part(n).x + 0.5, shape.part(n).y + 0.5] for n in range(shape.num_parts)])

    lm = np.round(lm) + 0.5

    lm_chin = lm[0: 17]  # left-right
    lm_eyebrow_left = lm[17: 22]  # left-right
    lm_eyebrow_right = lm[22: 27]  # left-right
    lm_nose = lm[27: 31]  # top-down
    lm_nostrils = lm[31: 36]  # top-down
    lm_eye_left = lm[36: 42]  # left-clockwise
    lm_eye_right = lm[42: 48]  # left-clockwise
    lm_mouth_outer = lm[48: 60]  # left-clockwise
    lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)

    eye_avg = (eye_left + eye_right) * 0.5

    # nose_mock_avg = (lm_nose[0] + lm_nose[1]) * 0.5
    # eye_avg = nose_mock_avg

    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)

    y = np.flipud(x) * [-1, 1]

    c = eye_avg + eye_to_mouth * 0.1

    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    crop = np.array(crop)
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(tuple(crop))
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(),
                        Image.BILINEAR)

    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)

    img = np.asarray(img)
    img = np.expand_dims(img, axis=0)
    img = np.float32(img) / 255

    return img


def gaussian_image(size, sigma, dim=2):
    if isinstance(size, numbers.Number):
        size = [size] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    weight_kernel = 1
    meshgrids = np.meshgrid(*[np.arange(size, dtype=np.float32) for size in size])
    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    for size, std, mgrid in zip(size, sigma, meshgrids):
        mean = (size - 1) / 2
        weight_kernel *= 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-((mgrid - mean) / std) ** 2 / 2)

    weight_kernel = weight_kernel / np.sum(weight_kernel)
    return weight_kernel


def inverse_gaussian_image(size, sigma, dim=2):
    gauss = gaussian_image(size, sigma, dim)

    # Inversion achieved by max - gauss, but adding min as well to
    # prevent regions of zeros which don't exist in normal gaussian
    inv_gauss = np.max(gauss) + np.min(gauss) - gauss
    inv_gauss = inv_gauss / np.sum(inv_gauss)

    return inv_gauss


def is_float(tensor):
    """
    Check if input tensor is float32, tensor maybe tf.Tensor or np.array
    """

    return (isinstance(tensor, tf.Tensor) and tensor.dtype != tf.dtypes.uint8) or tensor.dtype != np.uint8


def convert_tensor_to_image(tensor):
    """
    Converts tensor to image, and saturate output's range
    :param tensor: tf.Tensor, dtype float32, range [0,1]
    :return: np.array, dtype uint8, range [0, 255]
    """
    if is_float(tensor):
        tensor = tf.clip_by_value(tensor, 0., 1.)
        tensor = 255 * tensor

    if tensor.ndim == 4 and tensor.shape[0] == 1:
        tensor = tf.squeeze(tensor)

    tensor = np.uint8(np.round(tensor))

    return tensor


def save_image(img, file_path):
    """
    :param img: Could be either tf tensor or numpy array
    :param file_path:
    """

    if isinstance(file_path, Path):
        file_path = str(file_path)

    img = convert_tensor_to_image(img)
    img = Image.fromarray(img)
    img.save(file_path)


def mark_landmarks(img, lnd, color=None):
    """
    landmarks in (x,y) format
    """
    img = convert_tensor_to_image(img)
    radius = int(img.shape[0] / 256)

    lnd = (img.shape[0] / 160) * lnd

    if not color:
        color = (255, 255, 255)

    for i in range(lnd.shape[0]):
        x_y = lnd[i]
        img = cv2.circle(img, center=(int(x_y[0]), int(x_y[1])),
                         color=color, radius=radius, thickness=-1)

    return img


def get_weights(slope=0.2):
    """
    The scale is calculated according to:
        https://pytorch.org/docs/stable/nn.init.html
    and
        https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79

    For ReLU and LeakyReLU activations, the preferable initialization is kaiming.

    In Pytorch, the gain for LeakyReLU is calcaulted by: sqrt(2 / ( 1 + leaky_relu_slope ^ 2)) and the weights are
    sampled from N(0, std^2) where std = gain / sqrt(fan_in)

    To mimic this in TF, I am using VarianceScaling. The weights are sampled from N(0, std^2)
    where std = sqrt(scale / fan_in). Therefore, scale = gain^2
    """
    scale = 2 / (1 + slope ** 2)
    return VarianceScaling(scale)


def np_permute(tensor, permute):
    idx = np.empty_like(permute)
    idx[permute] = np.arange(len(permute))
    return tensor[:, idx]
