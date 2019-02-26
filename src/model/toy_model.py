"""Simple generator and discriminator models.

Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""

import tensorflow as tf
from core import CoreModelTPU

def _leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=0.0)


def _batch_norm(x, is_training, name):
  return tf.layers.batch_normalization(
      x, momentum=0.9, epsilon=1e-5, training=is_training, name=name)


class ToyModel(CoreModelTPU):
    """
    Example definition of a model/network architecture using this template.
    """

    def discriminator(self, x, is_training=True, scope='Discriminator', noise_dim=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            tf.logging.debug('Discriminator %s', self.dataset)
            tf.logging.debug('D -- Input %s', x)

            x = tf.layers.Dense(units=50, name='d_fc_1')(x)
            x = _leaky_relu(_batch_norm(x, is_training, name='d_bn1'))

            x = tf.layers.Dense(units=200, name='d_fc_2')(x)
            x = _leaky_relu(_batch_norm(x, is_training, name='d_bn2'))

            discriminate = tf.layers.Dense(units=1, name='discriminate')(x)
            tf.logging.debug(discriminate)

            if noise_dim:
                encode = tf.layers.Dense(units=noise_dim, name='encode')(x)
                tf.logging.debug(encode)
                return discriminate, encode

            return discriminate


    def generator(self, x, is_training=True, scope='Generator'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            tf.logging.debug('Generator %s', self.dataset)
            tf.logging.debug('G -- Input %s', x)

            tf.layers.Dense(units=200, name='g_fc_1')(x)
            x = _leaky_relu(_batch_norm(x, is_training, name='g_bn1'))

            tf.layers.Dense(units=50, name='g_fc_1')(x)
            x = _leaky_relu(_batch_norm(x, is_training, name='g_bn1'))

            x = tf.layers.Dense(units=2, name='fc_3')(x)

            return x