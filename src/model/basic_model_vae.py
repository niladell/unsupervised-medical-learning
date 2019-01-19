"""Simple generator and discriminator models.

Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""

import tensorflow as tf
from core import CoreModelTPU_VAE

def _leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=0.2)


def _batch_norm(x, is_training, name):
  return tf.layers.batch_normalization(
      x, momentum=0.9, epsilon=1e-5, training=is_training, name=name)


def _conv2d(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)


def _deconv2d(x, filters, kernel_size, stride, name):
  return tf.layers.conv2d_transpose(
      x, filters, [kernel_size, kernel_size],
      strides=[stride, stride], padding='same',
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)


class BasicModelVAE(CoreModelTPU_VAE):
    """
    Example definition of a model/network architecture using this template.
    """

    def discriminator(self, x, is_training=True, scope='Discriminator', code_dim=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            tf.logging.debug('Discriminator %s', self.dataset)
            tf.logging.debug('D -- Input %s', x)

            df_dim = 64  # Still not sure of this:
                         # Form carpedm20 "Dimension of gen filters in first conv layer"

            if self.dataset == 'cq500':

                # 512 x 512
                x = _conv2d(x, df_dim, 5, 2, name='d_conv0')
                x = _leaky_relu(x)
                tf.logging.debug(x)

                # 256 x 256
                x = _conv2d(x, df_dim, 5, 2, name='d_conv1.1')
                x = _leaky_relu(_batch_norm(x, is_training, name='d_bn1.1'))
                tf.logging.debug(x)

                # 128 x 128
                x = _conv2d(x, df_dim, 5, 2, name='d_conv1.2')
                x = _leaky_relu(_batch_norm(x, is_training, name='d_bn1.2'))
                tf.logging.debug(x)

                # 64 x 64
                x = _conv2d(x, df_dim * 2, 5, 2, name='d_conv1')
                x = _leaky_relu(_batch_norm(x, is_training, name='d_bn1'))
                tf.logging.debug(x)

                # 32 x 32
                x = _conv2d(x, df_dim * 2, 5, 2, name='d_conv4')
                x = _leaky_relu(_batch_norm(x, is_training, name='d_bn4'))
                tf.logging.debug(x)


            elif self.dataset == 'celebA':
                # 64 x 64
                x = _conv2d(x, df_dim, 5, 2, name='d_conv0')
                x = _leaky_relu(x)
                tf.logging.debug(x)

                # 32 x 32
                x = _conv2d(x, df_dim * 2, 5, 2, name='d_conv1')
                x = _leaky_relu(_batch_norm(x, is_training, name='d_bn1'))
                tf.logging.debug(x)

            elif self.dataset == 'CIFAR10':
                # 32 x 32
                x = _conv2d(x, df_dim * 2, 5, 2, name='d_conv1')
                x = _leaky_relu(x)
                tf.logging.debug(x)

            '''
            # 16 x 16
            x = _conv2d(x, df_dim * 4, 5, 2, name='d_conv2')
            x = _leaky_relu(_batch_norm(x, is_training, name='d_bn2'))
            tf.logging.debug(x)

            # 8 x 8
            x = _conv2d(x, df_dim * 8, 5, 2, name='d_conv3')
            x = _leaky_relu(_batch_norm(x, is_training, name='d_bn3'))
            tf.logging.debug(x)
            '''

            # Reshaping
            x = tf.reshape(x, [-1, 16 * 16 * df_dim * 8])
            tf.logging.debug(x)

            x = layers.Dense(2*code_dim, activation='relu')(x)

            z_mu = tf.layers.Dense(units=code_dim, name='encode')(x)
            z_log_sigma = tf.layers.Dense(units=code_dim, name='encode')(x)

            tf.logging.debug(z_mu)
            tf.logging.debug(z_log_sigma)

            return z_mu, z_log_sigma

    def sampling(args):
        z_mu, z_log_sigma = args
        epsilon = random_normal(shape=(shape(z_mu)[0], latent_dim),mean=0., stddev=1.)
        return z_mu + exp(z_log_sigma) * epsilon
    '''
    # sample vector from the latent distribution
    z = layers.Lambda(sampling)([z_mu, z_log_sigma])
    '''

    def generator(self, x, is_training=True, scope='Generator'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            tf.logging.debug('Generator %s', self.dataset)
            tf.logging.debug('G -- Input %s', x)

            gf_dim = 64  #   Still not sure of this:
                         # From carpedm20 "Dimension of gen filters in first conv layer"


            x = tf.layers.Dense(units=gf_dim * 8 * 16 * 16, name='g_fc1')(x)
            x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn1'))
            tf.logging.debug(x)
            x = tf.reshape(x, [-1, 16, 16, gf_dim * 8])
            tf.logging.debug(x)
            # 16 x 16

            '''
            x = _deconv2d(x, gf_dim * 4, 5, 2, name='g_dconv2')
            x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn2'))
            tf.logging.debug(x)
            # 8 x 8

            x = _deconv2d(x, gf_dim * 2, 4, 2, name='g_dconv3')
            x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn3'))
            tf.logging.debug(x)
            # 16 x 16
            '''

            if self.dataset == 'celebA':
                x = _deconv2d(x, gf_dim, 4, 2, name='g_dconv4')
                x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn4'))
                tf.logging.debug(x)
                # 32 x 32

                x = _deconv2d(x, 3, 4, 2, name='g_dconv5')
                tf.logging.debug(x)

            if self.dataset == 'cq500':
                x = _deconv2d(x, gf_dim, 4, 2, name='g_dconv4')
                x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn4'))
                tf.logging.debug(x)
                # 32 x 32

                x = _deconv2d(x, 1, 4, 2, name='g_dconv5')
                tf.logging.debug(x)
                # 64 x 64

                x = _deconv2d(x, 1, 4, 2, name='g_dconv6')
                tf.logging.debug(x)
                # 128 x 128

                x = _deconv2d(x, 1, 4, 2, name='g_dconv7')
                tf.logging.debug(x)
                # 256 x 256

                x = _deconv2d(x, 1, 4, 2, name='g_dconv8')
                tf.logging.debug(x)
                # 512 x 512
            else:
                x = _deconv2d(x, 3, 4, 2, name='g_dconv4')
                tf.logging.debug(x)
            x = tf.tanh(x)
            tf.logging.debug(x)

            return x