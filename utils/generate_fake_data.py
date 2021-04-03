import sys
from pathlib import Path
import os

sys.path.append('..')

import argparse

import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils.general_utils import save_image
from model.stylegan import StyleGAN_G


def main(args):
    base_dir = Path(args.output_path).joinpath(f'dataset_{args.resolution}')

    base_w_dir = base_dir.joinpath('ws')
    base_w_dir.mkdir(parents=True, exist_ok=True)

    base_im_dir = base_dir.joinpath('images')
    base_im_dir.mkdir(parents=True, exist_ok=True)

    existing_files = list(base_dir.joinpath('images').iterdir())
    if existing_files:
        max_exist = max([int(x.name) for x in existing_files])
        max_exist = int(max_exist - max_exist % 1e3 + 1e3)
    else:
        max_exist = 0

    stylegan_G_path = args.pretrained_models_path.joinpath(f'stylegan_G_{args.resolution}x{args.resolution}.h5')
    stylegan_G = StyleGAN_G(resolution=args.resolution, truncation_psi=args.truncation)
    stylegan_G.load_weights(str(stylegan_G_path))

    num_samples = args.num_images
    batch_size = args.batch_size
    num_batches = int(num_samples / batch_size)

    curr_ind = max_exist
    for _ in tqdm(range(num_batches)):
        z = tf.random.normal((batch_size, 512))
        w = stylegan_G.model_mapping(z)
        images = stylegan_G.model_synthesis(w)
        images = (images + 1) / 2

        if curr_ind % 1000 == 0:
            curr_w_dir = base_w_dir.joinpath(f'{curr_ind:05d}')
            curr_w_dir.mkdir(exist_ok=True)

            curr_im_dir = base_im_dir.joinpath(f'{curr_ind:05d}')
            curr_im_dir.mkdir(exist_ok=True)

        for j in range(batch_size):
            w_path = curr_w_dir.joinpath(f'{curr_ind:05d}.npy')
            np.save(str(w_path), w[j], allow_pickle=False)

            im_path = curr_im_dir.joinpath(f'{curr_ind:05d}.png')
            save_image(images[j], im_path)

            curr_ind += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--resolution', type=int, choices=[256, 1024], default=256)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--truncation', type=float, default=0.7)

    parser.add_argument('--output_path', required=True)
    parser.add_argument('--pretrained_models_path', type=Path, required=True)

    parser.add_argument('--num_images', type=int, default=10000)
    parser.add_argument('--gpu', default='0')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.num_images % 1e3 == 0

    main(args)
