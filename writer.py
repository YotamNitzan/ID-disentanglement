from utils.general_utils import convert_tensor_to_image
from pathlib import Path

import tensorflow as tf


class Writer(object):
    writer = None

    @staticmethod
    def set_writer(results_dir):
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)
        results_dir.mkdir(exist_ok=True, parents=True)
        Writer.writer = tf.summary.create_file_writer(str(results_dir))

    @staticmethod
    def add_scalar(tag, val, step):
        with Writer.writer.as_default():
            tf.summary.scalar(tag, val, step=step)

    @staticmethod
    def add_image(tag, val, step):
        val = convert_tensor_to_image(val)

        if tf.rank(val) == 3:
            val = tf.expand_dims(val, 0)

        with Writer.writer.as_default():
            tf.summary.image(tag, val, step)

    @staticmethod
    def flush():
        with Writer.writer.as_default():
            Writer.writer.flush()