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

ALPHA = -1.127

###############################################
# DEFINING THE INPUT FUNCTIONS FOR CQ500      #
###############################################

def input_fn(params):
    """Read CQ500 input data from a TFRecord dataset.

    Function taken from tensorflow/tpu cifar_keras repo
    """
    with tf.variable_scope('Input/input_fn'):
        batch_size = params['batch_size']
        data_dir = params['data_dir']
        noise_dim = params['noise_dim']

        def parser(serialized_example):
            """Parses a single tf.Example into image and label tensors."""

            # Parse the serialized data so we get a dict with our data.
            features = tf.parse_single_example(
                serialized_example,
                features={
                    "height": tf.FixedLenFeature([], tf.int64),
                    "width": tf.FixedLenFeature([], tf.int64),
                    "image": tf.FixedLenFeature([], tf.string),
                })

            # Decode the raw bytes so it becomes a tensor with type.
            image = tf.decode_raw(features["image"], tf.float32)

            # Hard-code the shape
            image.set_shape([CHANNELS * HEIGHT * WIDTH])

            # The type is now uint8 but we need it to be float.
            image = tf.reshape(image, [HEIGHT, WIDTH, CHANNELS]) * 2 - 1

            if params['noise_cov'].upper() == 'IDENTITY':
                random_noise = tf.random_normal([noise_dim])
            elif params['noise_cov'].upper() == 'POWER':
                x = tf.range(1, noise_dim+1, dtype=tf.float32)
                y = 100*tf.pow(x, ALPHA)
                random_noise = tf.random_normal(
                                shape=[noise_dim],
                                mean=tf.zeros_like(y),
                                stddev=y)
            else:
                raise NameError('{} is not an implemented distribution'.format(params['noise_cov']))

            features = {
                'real_images': image,
                'random_noise': random_noise}
            return features, []

        # TODO we should use an eval dataset fINDEBFOor EVAL  # pylint: disable=fixme
        if data_dir.endswith('.tfrecords'):
            image_files = data_dir
        else:
            image_files = [os.path.join(data_dir, 'train.tfrecords')]
        tf.logging.info(image_files)
        dataset = tf.data.TFRecordDataset([image_files])
        dataset = dataset.map(parser, num_parallel_calls=batch_size) # find a function to be able to use it with many files.
        # https://www.tensorflow.org/guide/performance/datasets: If your data can fit into memory, use the cache
        # transformation to cache it in memory during the first epoch, so that subsequent epochs can avoid the
        # overhead associated with reading, parsing, and transforming it.
        # dataset = dataset.prefetch(4 * batch_size).cache().repeat()
        dataset = dataset.prefetch(4 * batch_size).repeat()
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
            random_noise = tf.constant(tf.random_normal([batch_size, noise_dim]), dtype=tf.float32)
        elif params['noise_cov'].upper() == 'POWER':
            x = tf.range(1, noise_dim+1, dtype=tf.float32)
            y = 100*tf.pow(x, ALPHA)
            random_noise = tf.constant(tf.random_normal(
                            shape=[batch_size, noise_dim],
                            mean=tf.zeros_like(y),
                            stddev=y), dtype=tf.float32)
        else:
            raise NameError('{} is not an implemented distribution'.format(params['noise_cov']))

        noise_dataset = tf.data.Dataset.from_tensors(
            {'random_noise': random_noise
            # tf.constant(
                # np.random.randn(batch_size, noise_dim), dtype=tf.float32)
            })
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

if __name__ == '__main__':

    data_dir = './'
    params = {
        'batch_size':100,
        'data_dir': './',
        'noise_dim': 10
    }

    features, labels = input_fn(params)
    print(features)
    print(labels)
