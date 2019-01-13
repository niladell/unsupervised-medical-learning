"""Simple generator and discriminator models.

Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""

import tensorflow as tf
from core import CoreModelTPU
from .resBlocks_ops import batch_norm, g_block, d_block, _leaky_relu, _conv2d

class ResModel(CoreModelTPU):
    """
    Example definition of a model/network architecture using this template.
    """

    def discriminator(self, x, is_training=True, scope='Discriminator', noise_dim=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            tf.logging.debug('Discriminator %s', self.dataset)
            tf.logging.debug('D -- Input %s', x)

            df_dim = 32 # See gf_dim

            if self.dataset.upper() == 'CQ500':
                # Input: 512 x 512
                x = d_block(x, filters=df_dim, name='d_block1')
                tf.logging.debug(x)
                # 256
                x = d_block(x, filters=df_dim * 2, name='d_block2')
                tf.logging.debug(x)
                # 128
                x = d_block(x, filters=df_dim * 2, name='d_block3')
                tf.logging.debug(x)
                # 64
                x = d_block(x, filters=df_dim * 4, name='d_block4')
                tf.logging.debug(x)
                # 32
                x = d_block(x, filters=df_dim * 8, name='d_block5')
                tf.logging.debug(x)
                # 16
                x = d_block(x, filters=df_dim * 8, name='d_block6')
                tf.logging.debug(x)
                # 8
                x = d_block(x, filters=df_dim * 16, name='d_block7')
                tf.logging.debug(x)

                # 4 x 4
                x = tf.reshape(x, [-1, 4 * 4 * 16 * df_dim])
                tf.logging.debug(x)

            else:
                # TODO Clean this spaguetti
                if self.dataset == 'celebA':
                    # 64 x 64
                    x = d_block(x, filters=df_dim, name='d_block1')
                    tf.logging.debug(x)

                else:  # CIFAR10
                    # 32 x 32
                    pass #?
                # 32 x 32
                x = d_block(x, filters=df_dim * 2, name='d_block2')
                tf.logging.debug(x)

                # 16 x 16
                x = d_block(x, filters=df_dim * 4, name='d_block3')
                tf.logging.debug(x)

                # 8 x 8
                x = d_block(x, filters=df_dim * 8, name='d_block4')
                tf.logging.debug(x)

                # 4 x 4
                x = tf.reshape(x, [-1, 4 * 4 * 8 * df_dim])
                tf.logging.debug(x)

            logit = tf.layers.Dense(1, name='d_fc')(x)
            tf.logging.debug(logit)

            if noise_dim:
                encode = tf.layers.Dense(units=noise_dim, name='encode')(x)
                tf.logging.debug(encode)
                return logit, encode

            return logit

    def generator(self, x, is_training=True, scope='Generator'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            tf.logging.debug('Generator %s', self.dataset)
            tf.logging.debug('G -- Input %s', x)

            gf_dim = 32  #   Still not sure of this:
                         # Form carpedm20 "Dimension of gen filters in first conv layer"

            if self.dataset.upper() == 'CQ500':
                x = tf.layers.Dense(units=gf_dim * 16 * 4 * 4 )(x)
                tf.logging.debug(x)
                x = tf.reshape(x, [-1, 4, 4, gf_dim * 16])
                tf.logging.debug(x)

                x = g_block(x, gf_dim * 16, is_training, 'g_block1')  # 8 * 8
                tf.logging.debug(x)
                x = g_block(x, gf_dim * 8, is_training, 'g_block2')  # 16 * 16
                tf.logging.debug(x)
                x = g_block(x, gf_dim * 8, is_training, 'g_block3')  # 32 * 32
                tf.logging.debug(x)
                x = g_block(x, gf_dim * 4, is_training, 'g_block4')  # 64 * 64
                tf.logging.debug(x)
                x = g_block(x, gf_dim * 4, is_training, 'g_block5')  # 128 * 128
                tf.logging.debug(x)
                x = g_block(x, gf_dim * 2, is_training, 'g_block6')  # 256 * 256
                tf.logging.debug(x)
                x = g_block(x, gf_dim * 1, is_training, 'g_block7')  # 512 * 512
                tf.logging.debug(x)

                x = tf.nn.relu(batch_norm(x, is_training, 'bn'))
                x = tf.layers.Conv2D(filters=1, kernel_size=3, padding='SAME', name='conv_last')(x)

            else:
                x = tf.layers.Dense(units=gf_dim * 8 * 4 * 4 )(x)
                tf.logging.debug(x)
                x = tf.reshape(x, [-1, 4, 4, gf_dim * 8])
                tf.logging.debug(x)

                # TODO for now using 64x64, wee need to make this dynamic for the different sizes: {28?, 32, 64, 128?, 515}
                # x = g_block(x, gf_dim * 16, is_training, 'g_block1')  # 8 * 8
                # tf.logging.debug(x)
                x = g_block(x, gf_dim * 8, is_training, 'g_block2')  # 16 * 16
                tf.logging.debug(x)
                x = g_block(x, gf_dim * 4, is_training, 'g_block3')  # 32 * 32
                tf.logging.debug(x)
                x = g_block(x, gf_dim * 2, is_training, 'g_block4')  # 64 * 64
                tf.logging.debug(x)
                x = g_block(x, gf_dim * 1, is_training, 'g_block5')  # 128 * 128
                tf.logging.debug(x)

                x = tf.nn.relu(batch_norm(x, is_training, 'bn'))
                x = tf.layers.Conv2D(filters=3, kernel_size=3, padding='SAME', name='conv_last')(x)
            tf.logging.debug(x)

            x = tf.tanh(x)
            tf.logging.debug(x)

            return x
