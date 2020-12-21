import math
import shutil
import logging
import argparse
from pathlib import Path
from abc import ABC, abstractmethod


class BaseArgs(ABC):
    def __init__(self):
        self.args = None
        self.parser = argparse.ArgumentParser()
        self.logger = logging.getLogger(self.__class__.__name__)

        self.add_args()
        self.parse()
        self.validate()
        self.process()
        self.str_args = self.log()

    @abstractmethod
    def add_args(self):
        # Hardware
        self.parser.add_argument('--gpu', type=str, default='0')

        # Model
        self.parser.add_argument('--face_detection', action='store_true')
        self.parser.add_argument('--resolution', type=int, default=256, choices=[256, 1024])
        self.parser.add_argument('--load_checkpoint')
        self.parser.add_argument('--pretrained_models_path', type=Path, required=True)

        BaseArgs.add_bool_arg(self.parser, 'const_noise')

        # Data
        self.parser.add_argument('--batch_size', type=int, default=6)
        self.parser.add_argument('--reals', action='store_true', help='Use real inputs')
        BaseArgs.add_bool_arg(self.parser, 'test_real_attr')

        # Log & Results
        self.parser.add_argument('name', type=str, help='Name under which run will be saved')
        self.parser.add_argument('--results_dir', type=str, default='../results')
        self.parser.add_argument('--log_debug', action='store_true')

        # Other
        self.parser.add_argument('--debug', action='store_true')

    def parse(self):
        self.args = self.parser.parse_args()

    def log(self):
        out_str = 'The arguments are:\n'
        for k, v in self.args.__dict__.items():
            out_str += f'{k}: {v}\n'

        return out_str

    @staticmethod
    def add_bool_arg(parser, name, default=True):
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true')
        group.add_argument('--no_' + name, dest=name, action='store_false')
        parser.set_defaults(**{name: default})

    @abstractmethod
    def validate(self):
        if self.args.load_checkpoint and not Path(self.args.load_checkpoint).exists():
            raise ValueError(f'Checkpoint directory {self.args.load_checkpoint} does not exist')

    @abstractmethod
    def process(self):
        # Log & Results
        self.args.results_dir = Path(self.args.results_dir).joinpath(self.args.name)

        if self.args.debug:
            self.args.log_debug = True
        if self.args.debug or not self.args.train:
            shutil.rmtree(self.args.results_dir, ignore_errors=True)

        self.args.results_dir.mkdir(parents=True, exist_ok=True)
        self.args.images_results = self.args.results_dir.joinpath('images')
        self.args.images_results.mkdir(exist_ok=True)

        # Model
        if self.args.load_checkpoint:
            self.args.load_checkpoint = Path(self.args.load_checkpoint)


class TrainArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def add_args(self):
        super().add_args()

        self.parser.add_argument('--dataset_path', type=str, default='../my_dataset')

        self.parser.add_argument('--num_epochs', type=int, default=math.inf)
        self.parser.add_argument('--cross_frequency', type=int, default=3,
                                 help='Once in how many epochs to perform cross-train epoch (0 for never)')

        self.parser.add_argument('--unified', action='store_true')

        # Data
        BaseArgs.add_bool_arg(self.parser, 'train_real_attr', default=False)
        self.parser.add_argument('--train_data_size', type=int, default=70000,
                                 help='How many images to use for training. Others are used as validation')

        # Losses
        BaseArgs.add_bool_arg(self.parser, 'id_loss')
        BaseArgs.add_bool_arg(self.parser, 'landmarks_loss')
        BaseArgs.add_bool_arg(self.parser, 'pixel_loss')
        BaseArgs.add_bool_arg(self.parser, 'W_D_loss')
        BaseArgs.add_bool_arg(self.parser, 'gp')

        self.parser.add_argument('--pixel_mask_type', choices=['uniform', 'gaussian'], default='gaussian')
        self.parser.add_argument('--pixel_loss_type', choices=['L1', 'mix'], default='mix')

        # Test During training
        self.parser.add_argument('--test_frequency', type=int, default=1000,
                                 help='Once in how many epochs to perform a test')
        self.parser.add_argument('--test_size', type=int, default=50,
                                 help='How many mini-batches should be used for a test')
        self.parser.add_argument('--not_improved_exit', type=int, default=math.inf,
                                 help='After how many not-improved test to exit')
        BaseArgs.add_bool_arg(self.parser, 'test_with_arcface')

    def validate(self):
        super().validate()
        if not Path(self.args.dataset_path).exists():
            raise ValueError(f'Dataset at path: {self.args.dataset_path} does not exist')

    def process(self):
        self.args.train = True

        super().process()

        # Dataset
        self.args.dataset_path = Path(self.args.dataset_path)

        self.args.weights_dir = self.args.results_dir.joinpath('weights')
        self.args.weights_dir.mkdir(exist_ok=True)
        backup_code_dir = self.args.results_dir.joinpath('code')
        code_dir = Path().cwd()
        shutil.copytree(code_dir, backup_code_dir)


class TestArgs(BaseArgs):
    def __init__(self):
        super().__init__()

    def add_args(self):
        super().add_args()
        self.parser.set_defaults(batch_size=1)

        self.parser.add_argument('--id_dir', type=Path)
        self.parser.add_argument('--attr_dir', type=Path)
        self.parser.add_argument('--output_dir', type=Path)
        self.parser.add_argument('--input_dir', type=Path)

        self.parser.add_argument('--real_id', action='store_true')
        self.parser.add_argument('--real_attr', action='store_true')
        BaseArgs.add_bool_arg(self.parser, 'loop_fake')

        self.parser.add_argument('--img_suffixes', type=list, default=['png', 'jpg', 'jpeg'])

        self.parser.add_argument('--test_func', type=str, choices=['infer_on_dirs', 'infer_pairs', 'interpolate'])

        self.parser.add_argument('--input', type=str)

    def validate(self):
        super().validate()

        # if not self.args.input:
        #     raise ValueError('Input needed for inference')
        # if not Path(self.args.input).exists():
        #     raise ValueError(f'Input {self.args.input} does not exist')

    def process(self):
        self.args.train = False

        super().process()

        self.args.output_dir.mkdir(exist_ok=True, parents=True)

        # self.args.input = Path(self.args.input)
        # Split frame sit alongside input, so not every run needs to preprocess
        # input_name = self.args.input.stem
        # self.args.extracted_frames_dir = self.args.input.parent.joinpath(f'{input_name}_frames')
        # self.args.extracted_frames_dir.mkdir(exist_ok=True)
