import tensorflow as tf
from tensorflow.layers import Conv2D, BatchNormalization, AveragePooling2D


def _leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=0.2)


def batch_norm(x, is_training, name):
  return tf.layers.batch_normalization(
      x, momentum=0.9, epsilon=1e-5, training=is_training, name=name)


def _dense(x, channels, name):
  return tf.layers.dense(
      x, channels,
      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
      name=name)


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

def upsample(x, n):
  """Upscales the width and height of the input vector by a factor of n."""
  # Using this approach the same as in subpixel convolutions?
  if n < 2:
    return x
  return tf.batch_to_space(tf.tile(x, [n**2, 1, 1, 1]), [[0, 0], [0, 0]], n)

def g_block(x, filters, is_training, name):
  """Residual Block

  For a fast overview see Big GAN paper. Based on self-attention GAN code.
  """
  with tf.variable_scope(name):
    x0 = x
    x = tf.nn.relu(batch_norm(x, is_training, 'bn0'))
    x = upsample(x, 2)
    x = Conv2D(filters=filters, kernel_size=3, padding='SAME', name='conv1')(x)
    x = tf.nn.relu(batch_norm(x, is_training, 'bn1'))
    x = Conv2D(filters=filters, kernel_size=3, padding='SAME', name='conv2')(x)

    x0 = upsample(x0, 2)
    x0 = Conv2D(filters=filters, kernel_size=1, padding='SAME', name='conv3')(x0)

    return x0 + x

def d_block(x, filters, name):
  with tf.variable_scope(name):
    x0 = x
    x = tf.nn.relu(x)
    x = Conv2D(filters, kernel_size=3, padding='SAME', name='conv1')(x)
    x = tf.nn.relu(x)
    x = Conv2D(filters, kernel_size=3, padding='SAME', name='conv2')(x)
    x = AveragePooling2D(pool_size=2, strides=2, padding='VALID', name='avg1')(x)

    x0 = Conv2D(filters, kernel_size=3, padding='SAME', name='conv3')(x0)
    x0 = AveragePooling2D(pool_size=2, strides=2, padding='VALID', name='avg1')(x0)

    return x0 + x