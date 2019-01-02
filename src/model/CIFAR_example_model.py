"""Simple generator and discriminator models.

Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""

import tensorflow as tf

from core import CoreModelTPU

def _leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=0.2)


def _batch_norm(x, is_training, name):
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

class ExampleModel(CoreModelTPU):
    """
    Example definition of a model/network architecture using this template.
    """

    def discriminator(self, x, is_training=True, scope='Discriminator'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = _conv2d(x, 64, 5, 2, name='d_conv1')
            x = _leaky_relu(x)

            x = _conv2d(x, 128, 5, 2, name='d_conv2')
            x = _leaky_relu(_batch_norm(x, is_training, name='d_bn2'))

            x = _conv2d(x, 256, 5, 2, name='d_conv3')
            x = _leaky_relu(_batch_norm(x, is_training, name='d_bn3'))

            x = tf.reshape(x, [-1, 4 * 4 * 256])

            x = _dense(x, 1, name='d_fc_4')

            return x

    def generator(self, x, is_training=True, scope='Generator'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x = _dense(x, 4096, name='g_fc1')
            x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn1'))

            x = tf.reshape(x, [-1, 4, 4, 256])

            x = _deconv2d(x, 128, 5, 2, name='g_dconv2')
            x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn2'))

            x = _deconv2d(x, 64, 4, 2, name='g_dconv3')
            x = tf.nn.relu(_batch_norm(x, is_training, name='g_bn3'))

            x = _deconv2d(x, 3, 4, 2, name='g_dconv4')
            x = tf.tanh(x)

            return x

    def generate_model_fn(self):
        def model_fn(features, labels, mode, params):
            """Constructs DCGAN from individual generator and discriminator networks."""
            del labels    # Unconditional GAN does not use labels

            if mode == tf.estimator.ModeKeys.PREDICT:
                ###########
                # PREDICT #
                ###########
                # Pass only noise to PREDICT mode
                random_noise = features['random_noise']
                predictions = {
                    'generated_images': self.generator(random_noise, is_training=False)
                }

                return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)

            # Use params['batch_size'] for the batch size inside model_fn
            batch_size = params['batch_size']   # pylint: disable=unused-variable
            real_images = features['real_images']
            random_noise = features['random_noise']

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            generated_images = self.generator(random_noise,
                                                is_training=is_training)

            # Get logits from discriminator
            d_on_data_logits = tf.squeeze(self.discriminator(real_images))
            d_on_g_logits = tf.squeeze(self.discriminator(generated_images))

            # Calculate discriminator loss
            d_loss_on_data = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(d_on_data_logits),
                logits=d_on_data_logits)
            d_loss_on_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(d_on_g_logits),
                logits=d_on_g_logits)

            d_loss = d_loss_on_data + d_loss_on_gen

            # Calculate generator loss
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(d_on_g_logits),
                logits=d_on_g_logits)

            if mode == tf.estimator.ModeKeys.TRAIN:
                #########
                # TRAIN #
                #########
                d_loss = tf.reduce_mean(d_loss)
                g_loss = tf.reduce_mean(g_loss)
                d_optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.5)
                g_optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate, beta1=0.5)

                if self.use_tpu:
                    d_optimizer = tf.contrib.tpu.CrossShardOptimizer(d_optimizer)
                    g_optimizer = tf.contrib.tpu.CrossShardOptimizer(g_optimizer)

                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    d_step = d_optimizer.minimize(
                        d_loss,
                        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='Discriminator'))
                    g_step = g_optimizer.minimize(
                        g_loss,
                        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='Generator'))

                    increment_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
                    joint_op = tf.group([d_step, g_step, increment_step])

                return tf.contrib.tpu.TPUEstimatorSpec(
                        mode=mode,
                        loss=g_loss,
                        train_op=joint_op)

            elif mode == tf.estimator.ModeKeys.EVAL:
                ########
                # EVAL #
                ########
                def _eval_metric_fn(d_loss, g_loss):
                # When using TPUs, this function is run on a different machine than the
                # rest of the model_fn and should not capture any Tensors defined there
                    return {
                        'discriminator_loss': tf.metrics.mean(d_loss),
                        'generator_loss': tf.metrics.mean(g_loss)
                        }

                return tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=tf.reduce_mean(g_loss),
                    eval_metrics=(_eval_metric_fn, [d_loss, g_loss]))

            # Should never reach here
            raise ValueError('Invalid mode provided to model_fn')
        return model_fn
