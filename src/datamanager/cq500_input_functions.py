import os
import numpy as np

import tensorflow as tf
USE_ALTERNATIVE = False
try:
    from tensorflow.data import Dataset
except ImportError:
    tf.logging.warning('Using alternative settings due to old TF version')
    Dataset = tf.data.Dataset
    USE_ALTERNATIVE = True


HEIGHT = 512
WIDTH = 512
CHANNELS = 1



###############################################
# DEFINING THE INPUT FUNCTIONS FOR CQ500      #
###############################################

def input_fn(params):
    """Read CQ500 input data from a TFRecord dataset.

    Function taken from tensorflow/tpu cifar_keras repo"""
    batch_size = params['batch_size']
    data_dir = params['data_dir']
    noise_dim = params['noise_dim']

    def parser(serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                "height": tf.FixedLenFeature([], tf.int64),
                "width": tf.FixedLenFeature([], tf.int64),
                "image_raw": tf.FixedLenFeature([], tf.string),
            })
        image = tf.decode_raw(features["image"], tf.uint8)
        image.set_shape([CHANNELS * HEIGHT * WIDTH])

        image = tf.cast(
                    tf.reshape(image, [HEIGHT, WIDTH, CHANNELS]),
                tf.float32) * (2. / 255) - 1
        random_noise = tf.random_normal([noise_dim])
        features = {
            'real_images': image,
            'random_noise': random_noise}

        return features, []

    # TODO we should use an eval dataset fINDEBFOor EVAL  # pylint: disable=fixme
    image_files = [os.path.join(data_dir, 'train.tfrecords')]
    tf.logging.info(image_files)
    dataset = tf.data.TFRecordDataset([image_files])
    dataset = dataset.map(parser, num_parallel_calls=batch_size) # find a function to be able to use it with many files.
    dataset = dataset.prefetch(4 * batch_size).cache().repeat()
    if USE_ALTERNATIVE:
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
        tf.logging.warning('Old version: Used \
           tf.contrib.data.batch_and_drop_remainder instead of regular batch')
    else:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    # Not sure why we use one_shot and not initializable_iterator
    features, labels = dataset.make_one_shot_iterator().get_next()
    tf.logging.debug('Input_fn: Features %s, Labels %s', features, labels)
    return features, labels


def noise_input_fn(params):
    """Input function for generating samples for PREDICT mode.

    Generates a single Tensor of fixed random noise. Use tf.data.Dataset to
    signal to the estimator when to terminate the generator returned by
    predict().

    Args:
        params: param `dict` passed by TPUEstimator.

    Returns:
        1-element `dict` containing the randomly generated noise.
    """
    batch_size = params['batch_size']
    noise_dim = params['noise_dim']
    # Use constant seed to obtain same noise
    np.random.seed(0)
    noise_dataset = tf.data.Dataset.from_tensors(tf.constant(
        np.random.randn(batch_size, noise_dim), dtype=tf.float32))
    noise = noise_dataset.make_one_shot_iterator().get_next()
    tf.logging.debug('Noise input %s', noise)
    return {'random_noise': noise}, None


def generate_input_fn(mode='TRAIN'):
    """Creates input_fn depending on whether the code is training or not."""
    mode = mode.upper()
    if mode == 'TRAIN' or mode == 'EVAL':
        return input_fn
    elif mode == 'PREDICT' or mode == 'NOISE':
        return noise_input_fn
    else:
        raise ValueError('Incorrect mode provided')

filename = 'CT_head_trauma_unzipped_CQ500CT0 CQ500CT0_Unknown Study_CT PLAIN THIN_CT000000.dcm.tfrecords'
reader = tf.TFRecordReader()
reader = tf.data.TFRecordDataset()
reader.read(filename)

x = tf.constant("This is string")
x = tf.decode_raw(x, tf.uint8)
y = x[:4]
sess = tf.InteractiveSession()
print(y.eval())

