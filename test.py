import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['USE_SIMPLE_THREADED_LEVEL3'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import sys
from model.stylegan import StyleGAN_G_synthesis
from model.model import Network
from writer import Writer
from inference import Inference
from arglib import arglib
import utils

sys.path.insert(0, 'model/face_utils')



def main():
    test_args = arglib.TestArgs()
    args, str_args = test_args.args, test_args.str_args
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    Writer.set_writer(args.results_dir)

    id_model_path = args.pretrained_models_path.joinpath('vggface2.h5')
    stylegan_G_synthesis_path = str(
        args.pretrained_models_path.joinpath(f'stylegan_G_{args.resolution}x{args.resolution}_synthesis'))

    utils.landmarks_model_path = str(args.pretrained_models_path.joinpath('shape_predictor_68_face_landmarks.dat'))

    stylegan_G_synthesis = StyleGAN_G_synthesis(resolution=args.resolution, is_const_noise=args.const_noise)
    stylegan_G_synthesis.load_weights(stylegan_G_synthesis_path)

    network = Network(args, id_model_path, stylegan_G_synthesis)

    network.test()
    inference = Inference(args, network)
    test_func = getattr(inference, args.test_func)
    test_func()


if __name__ == '__main__':
    main()
