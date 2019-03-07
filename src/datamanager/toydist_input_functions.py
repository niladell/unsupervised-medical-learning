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

# NUMBER_OF_GAUSSIANS = 1

# def sample_toy_distr(number_of_gaussians):
#     """ Grid layed gaussians"""
#     x = np.random.normal(0, 0.05)
#     y = np.random.normal(0, 0.05)
#     # centers = [(0,-1), (1,0), (-0.5,0.5)]
#     centers = np.vstack([[i,j] for i in range(number_of_gaussians) for j in range(number_of_gaussians)])
#     centers = centers - np.mean(centers, axis=0)

#     center = np.random.randint(0, centers.shape[0])
#     mu_x, mu_y = centers[center]
#     return [x + mu_x, y + mu_y]

def sample_toy_distr(number_of_gaussians):
    """ Circle layed gaussians   """

    x = np.random.normal(0, 0.02)
    y = np.random.normal(0, 0.02)
    # centers = [(0,-1), (1,0), (-0.5,0.5)]
    # centers = np.vstack([[i,j] for i in range(number_of_gaussians) for j in range(number_of_gaussians)])
    centers = np.array([i for i in range(number_of_gaussians)])
    centers = 2*np.pi*centers/np.max(centers + 1e-8)
    center = np.random.randint(0, centers.shape[0])
    c = centers[center]
    cx = np.cos(c)*2
    cy = np.sin(c)*2

    return np.array([x + cx, y + cy])

def input_fn(params):
    """Read CIFAR input data from a TFRecord dataset.

    Function taken from tensorflow/tpu cifar_keras repo"""
    batch_size = params['batch_size']
    data_dir = params['data_dir']
    noise_dim = params['noise_dim']
    assert 'number_of_gaussians' in params
    number_of_gaussians = params['number_of_gaussians']

    def gen():
        """Parses a single tf.Example into image and label tensors."""
        for _ in range(100000):
            data = sample_toy_distr(number_of_gaussians)
            random_noise = np.random.normal(np.zeros(noise_dim))

            yield data, random_noise, 0

    # TODO we should use an eval dataset fINDEBFOor EVAL  # pylint: disable=fixme
    image_files = [os.path.join(data_dir, 'train.tfrecords')]
    tf.logging.info(image_files)
    dataset = tf.data.Dataset.from_generator(
                                gen,
                                output_types=(tf.float32, tf.float32, tf.float32),
                                output_shapes=(tf.TensorShape([2]), tf.TensorShape([noise_dim]), tf.TensorShape([]))
                            )
    # dataset = dataset.prefetch(4 * batch_size).cache()#.repeat()
    if USE_ALTERNATIVE:
        dataset = dataset.apply(
            tf.contrib.data.batch_and_drop_remainder(batch_size))
        tf.logging.warning('Old version: Used \
           tf.contrib.data.batch_and_drop_remainder instead of regular batch')
    else:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    # Not sure why we use one_shot and not initializable_iterator
    toy_dist, random_noise, labels = dataset.make_one_shot_iterator().get_next()
    # toy_dist.set_shape([batch_size, 2])
    # random_noise.set_shape([batch_size, noise_dim])
    features = {
            'real_images' : toy_dist,
            'random_noise': random_noise}

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
