import logging

import numpy as np
import tensorflow as tf

from utils.general_utils import read_image


class DataLoader(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)

        self.real_dataset = args.dataset_path.joinpath(f'real')

        dataset = args.dataset_path.joinpath(f'dataset_{args.resolution}')

        self.ws_dataset = dataset.joinpath('ws')
        self.image_dataset = dataset.joinpath('images')

        max_dir = max([x.name for x in self.image_dataset.iterdir()])
        self.max_ind = max([int(x.stem) for x in self.image_dataset.joinpath(max_dir).iterdir()])
        self.train_max_ind = args.train_data_size

        if self.train_max_ind >= self.max_ind:
            self.logger.warning('There is no validation data... using training data')
            self.min_val_ind = 0
            self.train_max_ind = self.max_ind
        else:
            self.min_val_ind = self.train_max_ind + 1

    def get_image(self, is_train, black_list=None, is_real=False):
        # Default should be non-mutable
        if black_list is None:
            black_list = []

        max_fails = 10
        curr_fail = 0
        if is_train:
            min_ind, max_ind = 0, self.train_max_ind
        else:
            min_ind, max_ind = self.min_val_ind, self.max_ind

        while True:
            ind = np.random.randint(min_ind, max_ind)

            if ind in black_list:
                continue

            img_name = f'{ind:05d}.png'
            dir_name = f'{int(ind - ind % 1e3):05d}'
            if is_real:
                img_path = self.real_dataset.joinpath(dir_name, img_name)
            else:
                img_path = self.image_dataset.joinpath(dir_name, img_name)

            try:
                img = read_image(img_path, self.args.resolution)
                break
            except Exception as e:
                self.logger.warning(f'Failed reading image at {ind}. Error: {e}')

                # Try again with a different image...
                curr_fail += 1
                if curr_fail > max_fails:
                    raise IOError('Failed reading multiples images')
                continue

        return ind, img

    def get_w_by_ind(self, ind):
        dir_name = f'{int(ind - ind % 1e3):05d}'
        img_name = f'{ind:05d}.npy'
        w_path = self.ws_dataset.joinpath(dir_name, img_name)

        w = np.load(w_path)

        # Take one row while keeping dimension
        w = w[np.newaxis, 0]

        return w

    def get_real_w(self, is_train, black_list=None, is_real=False):
        ind = np.random.randint(0, self.max_ind)
        w = self.get_w_by_ind(ind)

        return ind, w

    def batch_samples(self, get_sample_func, is_train, black_list=None, is_real=False):
        batch = []
        indices = []

        if not black_list:
            black_list = []
        for i in range(self.args.batch_size):
            ind, sample = get_sample_func(is_train, black_list, is_real)

            batch.append(sample)
            indices.append(ind)

        batch = tf.concat(batch, 0)

        return indices, batch

    def get_batch(self, is_train=True, is_cross=False, ws=True):
        black_list = []
        id_imgs_indices, id_img = self.batch_samples(self.get_image, is_train)
        matching_ws = None

        self.logger.debug(f'ID images read: {id_imgs_indices}')
        black_list.extend(id_imgs_indices)

        if is_cross:
            # Use real attr when args say so or when testing
            is_real_attr = (is_train and self.args.train_real_attr) or (not is_train and self.args.test_real_attr)
            black_list = [] if is_real_attr else black_list

            attr_imgs_indices, attr_img = self.batch_samples(self.get_image,
                                                             is_train,
                                                             black_list=black_list,
                                                             is_real=is_real_attr)

            self.logger.debug(f'Attr images read: {attr_imgs_indices}')

        else:
            if is_train:
                attr_img = id_img
                matching_ws = [self.get_w_by_ind(ind) for ind in id_imgs_indices]
                matching_ws = tf.concat(matching_ws, 0)
            else:
                attr_img = id_img

        if not is_train:
            return attr_img, id_img

        # Only for training
        real_img = None
        real_ws = None

        if self.args.train and self.args.reals:
            real_imgs_indices, real_img = self.batch_samples(self.get_image, is_train, black_list=[], is_real=True)
            self.logger.debug(f'Real images read: {real_imgs_indices}')

        if ws:
            _, real_ws = self.batch_samples(self.get_real_w, is_train)

        return attr_img, id_img, real_ws, real_img, matching_ws

