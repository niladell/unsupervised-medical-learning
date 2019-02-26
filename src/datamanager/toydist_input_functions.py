import os
import numpy as np
import random

import tensorflow as tf
USE_ALTERNATIVE = False
try:
    from tensorflow.data import Dataset
except ImportError:
    tf.logging.warning('Using alternative settings due to old TF version')
    Dataset = tf.data.Dataset
    USE_ALTERNATIVE = True

def sample_toy_distr():
    x = np.random.normal(0, 1)
    y = np.random.normal(0, 1)
    centers = [(0,0) , (0,2),  (2,0), (2,2), (1,1)]
    mu_x, mu_y = random.sample(centers,1)[0]
    return [x + mu_x, y + mu_y]

def input_fn(params):
    """Read CIFAR input data from a TFRecord dataset.

    Function taken from tensorflow/tpu cifar_keras repo"""
    batch_size = params['batch_size']
    data_dir = params['data_dir']
    noise_dim = params['noise_dim']

    def gen():
        """Parses a single tf.Example into image and label tensors."""
        # features = {
        #     'real_images' : sample_toy_distr(),
        #     'random_noise': random_noise}
        # data = tf.constant(sample_toy_distr(), dtype=tf.float32)
        # data = tf.reshape(data, (2))
        # data.set_shape([2])

        # random_noise = tf.random_normal([noise_dim])
        # random_noise = tf.reshape(random_noise, (noise_dim))
        # random_noise.set_shape(noise_dim)

        data = sample_toy_distr()
        random_noise = np.random.normal(np.zeros(noise_dim))

        yield data, random_noise, []

    # TODO we should use an eval dataset fINDEBFOor EVAL  # pylint: disable=fixme
    image_files = [os.path.join(data_dir, 'train.tfrecords')]
    tf.logging.info(image_files)
    dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.float32, tf.float32))
    dataset = dataset.prefetch(4 * batch_size).cache().repeat()
    if USE_ALTERNATIVE:
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
        tf.logging.warning('Old version: Used \
           tf.contrib.data.batch_and_drop_remainder instead of regular batch')
    else:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    # Not sure why we use one_shot and not initializable_iterator
    toy_dist, random_noise, labels = dataset.make_one_shot_iterator().get_next()
    toy_dist.set_shape([batch_size, 2])
    random_noise.set_shape([batch_size, noise_dim])
    features = {
            'real_images' : toy_dist,
            'random_noise': random_noise}

    tf.logging.debug('Input_fn: Features %s, Labels %s', features, labels)
    return features, labels

# def input_fn(params):
#     """Input function for generating samples for PREDICT mode.

#     Generates a single Tensor of fixed random noise. Use tf.data.Dataset to
#     signal to the estimator when to terminate the generator returned by
#     predict().

#     Args:
#         params: param `dict` passed by TPUEstimator.

#     Returns:
#         1-element `dict` containing the randomly generated noise.
#     """
#     batch_size = params['batch_size']
#     noise_dim = params['noise_dim']
#     dataset = tf.data.Dataset.from_tensor_slices(
#             (
#                 {
#                     'real_images' : tf.constant(toy_samples(batch_size), dtype=tf.float32),
#                     'random_noise': tf.constant(noise_vector(batch_size, noise_dim), dtype=tf.float32)
#                 },
#                 tf.constant([[]]*batch_size, dtype=tf.float32))
#         )

#     features, labels = dataset.make_one_shot_iterator().get_next()
#     tf.logging.debug('Input_fn: Features %s', features)
#     return features, labels

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
