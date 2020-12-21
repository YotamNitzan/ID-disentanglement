import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['USE_SIMPLE_THREADED_LEVEL3'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
import logging
from model.stylegan import StyleGAN_G_synthesis
from model.model import Network
from data_loader.data_loader import DataLoader
from writer import Writer
from trainer import Trainer
from arglib import arglib
from utils import general_utils as utils

sys.path.insert(0, 'model/face_utils')


def init_logger(args):
    root_logger = logging.getLogger()

    level = logging.DEBUG if args.log_debug else logging.INFO
    root_logger.setLevel(level)

    file_handler = logging.FileHandler(f'{args.results_dir}/log.txt')
    console_handler = logging.StreamHandler()

    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt)

    file_handler.setLevel(level)
    console_handler.setLevel(level)

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    pil_logger = logging.getLogger('PIL.PngImagePlugin')
    pil_logger.setLevel(logging.INFO)


def main():
    train_args = arglib.TrainArgs()
    args, str_args = train_args.args, train_args.str_args
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    init_logger(args)

    logger = logging.getLogger('main')

    cmd_line = ' '.join(sys.argv)
    logger.info(f'cmd line is: \n {cmd_line}')

    logger.info(str_args)
    logger.debug('Copying src to results dir')

    Writer.set_writer(args.results_dir)

    if not args.debug:
        description = input('Please write a short description of this run\n')
        desc_file = args.results_dir.joinpath('description.txt')
        with desc_file.open('w') as f:
            f.write(description)

    id_model_path = args.pretrained_models_path.joinpath('vggface2.h5')
    stylegan_G_synthesis_path = str(
        args.pretrained_models_path.joinpath(f'stylegan_G_{args.resolution}x{args.resolution}_synthesis'))
    landmarks_model_path = str(args.pretrained_models_path.joinpath('face_utils/keypoints'))
    face_detection_model_path = str(args.pretrained_models_path.joinpath('face_utils/detector'))

    arcface_model_path = str(args.pretrained_models_path.joinpath('arcface_weights/weights-b'))
    utils.landmarks_model_path = str(args.pretrained_models_path.joinpath('shape_predictor_68_face_landmarks.dat'))

    stylegan_G_synthesis = StyleGAN_G_synthesis(resolution=args.resolution, is_const_noise=args.const_noise)
    stylegan_G_synthesis.load_weights(stylegan_G_synthesis_path)

    network = Network(args, id_model_path, stylegan_G_synthesis, landmarks_model_path,
                      face_detection_model_path, arcface_model_path)
    data_loader = DataLoader(args)

    trainer = Trainer(args, network, data_loader)
    trainer.train()


if __name__ == '__main__':
    main()
