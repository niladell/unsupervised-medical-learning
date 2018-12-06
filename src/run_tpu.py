from absl import flags
from absl import logging
# import coloredlogs
logging.set_verbosity(logging.INFO)
#coloredlogs.install(level='INFO')
from os import walk
import pickle
import numpy as np
import functools

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver #pylint: disable=E0611
tfgan = tf.contrib.gan
queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework

# from core import DataManager
# from model import ExampleModel
# from datamanager import CIFAR10

SAMPLE_NUM = 50000
CHANNELS = 3
SIDE = 32

BATCH_SIZE = 128
NOISE_DIMS = 64
### HIGHLY PROTOTIPY VERSON BELOW -- TODO MOVE TO DATA MANAGER

def input_fn():
    """Read CIFAR input data from a TFRecord dataset.
    
    Function taken from tensorflow/tpu cifar_keras repo"""
    # del params
    batch_size = BATCH_SIZE
    def parser(serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                "image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features["image"], tf.uint8)
        image.set_shape([3*32*32])
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image = tf.reshape(image, [32, 32, 3])
        label = tf.cast(features["label"], tf.int32)
        logging.info(image)
        noise = tf.random_normal([NOISE_DIMS])
        return noise, image #, label

    # TEMPORAL
    image_files = ['gs://iowa_bucket/cifar-10-data/train.tfrecords']

    dataset = tf.data.TFRecordDataset([image_files])
    dataset = dataset.map(parser, num_parallel_calls=batch_size)
    dataset = dataset.prefetch(4 * batch_size).cache().repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)
    return dataset

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def generator_fn(noise, weight_decay=2.5e-5, is_training=True):
    """Simple generator to produce MNIST images.

    Args:
        noise: A single Tensor representing noise.
        weight_decay: The value of the l2 weight decay.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population 
            statistics.

    Returns:
        A generated image in the range [-1, 1].
    """
    with framework.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)),\
    framework.arg_scope([layers.batch_norm], is_training=is_training,
                        zero_debias_moving_mean=True):
        logging.info(noise)
        net = layers.fully_connected(noise, 1024)
        logging.info(net)
        net = layers.fully_connected(net, 7 * 7 * 256)
        logging.info(net)
        net = tf.reshape(net, [-1, 7, 7, 256])
        logging.info(net)
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        logging.info(net)
        net = layers.conv2d_transpose(net, 32, [5, 5], stride=2, padding='VALID')
        logging.info(net)
        # TODO CHECK A PROPER WAY OF DOING THIS
        net = layers.conv2d_transpose(net, 32, [2, 2], stride=1, padding='VALID')
        logging.info(net)

        net = layers.conv2d(net, CHANNELS, 4, normalizer_fn=None, activation_fn=tf.tanh)
        logging.info(net)
    return net

def discriminator_fn(img, unused_conditioning, weight_decay=2.5e-5,
                is_training=True):
    """Discriminator network on MNIST digits.
    
    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        unused_conditioning: The TFGAN API can help with conditional GANs, which
            would require extra `condition` information to both the generator and the
            discriminator. Since this example is not conditional, we do not use this
            argument.
        weight_decay: The L2 weight decay.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population 
            statistics.
    
    Returns:
        Logits for the probability that the image is real.
    """
    with framework.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        with framework.arg_scope([layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(net, 1024, normalizer_fn=layers.batch_norm)
        return layers.linear(net, 1)


logging.info('Start')
noise_dims = 64
# Create GAN estimator.
gan_estimator = tfgan.estimator.GANEstimator(
    model_dir='gs://iowa_bucket/cifar10/outputs/',
    generator_fn=generator_fn,
    discriminator_fn=discriminator_fn,
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    generator_optimizer=tf.train.AdamOptimizer(0.1, 0.5),
    discriminator_optimizer=tf.train.AdamOptimizer(0.1, 0.5))

steps = 100000
gan_estimator.train(input_fn, steps=steps)

# model = ExampleModel(tf_session=None,
#                         learning_rate=0.001,
#                         data_dir= 'gs://iowa_bucket/cifar10/data/',  # Dataset in GCloud Bucket
#                         use_tpu=True,
#                         output_path='gs://iowa_bucket/cifar10/outputs/'
# )

logging.info('Build')
gan_estimator.build_model()




# ### END
# logging.info('Train')
# gan_estimator.train(10001, input_fn)
