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


HEIGHT = 64
WIDTH = 64
CHANNELS = 3

ALPHA = -1.43 # Reported one eigenfaces dimensionality paper


def input_fn(params):
    """Read CIFAR input data from a TFRecord dataset.

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
                "image": tf.FixedLenFeature([], tf.string),
            })
        image = tf.decode_raw(features["image"], tf.uint8)
        image.set_shape([CHANNELS * HEIGHT * WIDTH])

        image = tf.cast(
                    tf.reshape(image, [HEIGHT, WIDTH, CHANNELS]),
                tf.float32) * (2. / 255) - 1

        if params['noise_cov'].upper() == 'IDENTITY':
            random_noise = tf.random_normal([noise_dim], name='noise_generator')
        elif params['noise_cov'].upper() == 'POWER':
            x = tf.range(1, noise_dim+1, dtype=tf.float32)
            stdev = 100*tf.pow(x, ALPHA)
            random_noise = tf.random_normal(
                            shape=[noise_dim],
                            mean=tf.zeros(noise_dim),
                            stddev=stdev,
                            name='pnoise_generator')
        else:
            raise NameError('{} is not an implemented distribution'.format(params['noise_cov']))

        features = {
            'real_images': image,
            'random_noise': random_noise}

        return features, []

    # TODO we should use an eval dataset fINDEBFOor EVAL  # pylint: disable=fixme
    image_files = [os.path.join(data_dir, 'train.tfrecords')]
    tf.logging.info(image_files)
    dataset = tf.data.TFRecordDataset([image_files])
    dataset = dataset.map(parser, num_parallel_calls=batch_size)
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
    with tf.variable_scope('Input/noise_input'):
        batch_size = params['batch_size']
        noise_dim = params['noise_dim']
        # Use constant seed to obtain same noise
        np.random.seed(0)

        if params['noise_cov'].upper() == 'IDENTITY':
            random_noise = tf.constant(
                              np.random.randn(batch_size, noise_dim),
                            dtype=tf.float32, name='pred_noise_generator')
        elif params['noise_cov'].upper() == 'POWER':
            x = np.arange(1, noise_dim+1)
            stdev = 10*x**ALPHA
            eps = np.random.randn(batch_size, noise_dim)
            # This is the equivalent to the tf.random_normal used on top
            # see: https://github.com/tensorflow/tensorflow/blob/a6d8ffae097d0132989ae4688d224121ec6d8f35/tensorflow/python/ops/random_ops.py#L72-L81
            noise = eps * stdev
            random_noise = tf.constant(noise, dtype=tf.float32, name='pred_pnoise_generator')
        else:
            raise NameError('{} is not an implemented distribution'.format(params['noise_cov']))

        random_image = np.random.randn(batch_size, HEIGHT, WIDTH, CHANNELS).astype(np.float32)

        noise_dataset = tf.data.Dataset.from_tensors(
            {'random_noise': random_noise,
             'random_images': random_image })

        noise = noise_dataset.make_one_shot_iterator().get_next()
        tf.logging.debug('Noise input %s', noise)
        return noise_dataset


def generate_input_fn(mode='TRAIN'):
    """Creates input_fn depending on whether the code is training or not."""
    mode = mode.upper()
    if mode == 'TRAIN' or mode == 'EVAL':
        return input_fn
    elif mode == 'PREDICT' or mode == 'NOISE':
        return noise_input_fn
    else:
        raise ValueError('Incorrect mode provided')
