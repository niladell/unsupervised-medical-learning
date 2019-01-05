"""Simple generator and discriminator models.

Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""

import tensorflow as tf
from core import CoreModelTPU
from .vanilla_ops import batch_norm, g_block, d_block, _leaky_relu, _conv2d

class Model(CoreModelTPU):
    """
    Example definition of a model/network architecture using this template.
    """

    def discriminator(self, x, is_training=True, scope='Discriminator'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            tf.logging.debug('Discriminator %s', self.dataset)
            tf.logging.debug('D -- Input %s', x)

            df_dim = 32 # See gf_dim

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

            return logit

    def generator(self, x, is_training=True, scope='Generator'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            tf.logging.debug('Generator %s', self.dataset)
            tf.logging.debug('G -- Input %s', x)

            gf_dim = 32  #   Still not sure of this:
                         # Form carpedm20 "Dimension of gen filters in first conv layer"
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

    def generate_model_fn(self):

        def model_fn(features, labels, mode, params):
            """Constructs DCGAN from individual generator and discriminator
            networks.
            """
            del labels    # Unconditional GAN does not use labels
            if mode == tf.estimator.ModeKeys.PREDICT:
                ###########
                # PREDICT #
                ###########
                # Pass only noise to PREDICT mode
                random_noise = features['random_noise']
                predictions = {
                    'generated_images': self.generator(
                                            random_noise, is_training=False)
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
