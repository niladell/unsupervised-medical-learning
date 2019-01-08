"""Simple generator and discriminator models.

Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""

import tensorflow as tf
from core import CoreModelTPU


from .basic_nets import generator, discriminator
# from .resBlocks_net import generator, discriminator # TODO they are not yet have an encoder implemented
class Model(CoreModelTPU):
    """
    Example definition of a model/network architecture using this template.
    """

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
                    'generated_images': generator(
                                          random_noise, dataset=self.dataset, is_training=False)
                }

                return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)

            # Use params['batch_size'] for the batch size inside model_fn
            batch_size = params['batch_size']   # pylint: disable=unused-variable
            noise_dim = params['noise_dim']
            real_images = features['real_images']
            random_noise = features['random_noise']

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            generated_images = generator(random_noise,
                                          dataset=self.dataset, is_training=is_training)

            # Get logits from discriminator
            d_on_data_logits = tf.squeeze(discriminator(real_images, dataset=self.dataset))
            if self.use_encoder and self.encoder == 'ATTACHED':
                # If we use and embedded encoder we create it here
                d_on_g_logits, g_logits_encoded =\
                    discriminator(generated_images, dataset=self.dataset, noise_dim=noise_dim)
                d_on_g_logits = tf.squeeze(d_on_g_logits)
            else:
                # Regular GAN w/o encoder
                d_on_g_logits = tf.squeeze(discriminator(generated_images, dataset=self.dataset))

            # Create the labels
            true_label = tf.ones_like(d_on_data_logits)
            fake_label = tf.zeros_like(d_on_g_logits)
            #  We invert the labels for the generator training (ganTricks)
            true_label_g = tf.ones_like(d_on_g_logits)

            # Soften the labels (ganTricks)
            fuzzyness = 0.2
            if fuzzyness != 0:
                true_label += tf.random_uniform(true_label.shape,
                                            minval=-fuzzyness, maxval=fuzzyness)
                fake_label += tf.random_uniform(fake_label.shape,
                                            minval=-fuzzyness, maxval=fuzzyness)
                true_label_g += tf.random_uniform(true_label_g.shape,
                                            minval=-fuzzyness, maxval=fuzzyness)


            # Calculate discriminator loss
            d_loss_on_data = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_label,
                logits=d_on_data_logits)
            d_loss_on_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=fake_label,
                logits=d_on_g_logits)

            d_loss = d_loss_on_data + d_loss_on_gen

            # Calculate generator loss
            g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=true_label_g,
                logits=d_on_g_logits)

            # Create independent encoder
            if self.use_encoder:
                if self.encoder == 'INDEPENDENT':
                    _, g_logits_encoded = discriminator(generated_images,
                                                            dataset=self.dataset,
                                                            scope='Encoder',
                                                            noise_dim=noise_dim)
                e_loss = tf.losses.mean_squared_error(
                    labels=random_noise,
                    predictions=g_logits_encoded)


            if mode == tf.estimator.ModeKeys.TRAIN:
                #########
                # TRAIN #
                #########
                # TODO There has to be a less messy way of doing theis encoder steps
                if self.use_encoder and self.encoder == 'ATTACHED':
                    d_loss = d_loss + e_loss
                d_loss = tf.reduce_mean(d_loss)
                # Do we use the encoder loss to train on G or is it independent
                e_loss_on_g = True
                if self.use_encoder and e_loss_on_g:
                    g_loss = g_loss + e_loss
                g_loss = tf.reduce_mean(g_loss)
                # ? TODO is this the best way to deal with the optimziers?
                # d_optimizer = tf.train.GradientDescentOptimizer(
                #     learning_rate=self.learning_rate)
                # d_optimizer = tf.train.AdamOptimizer(
                #     learning_rate=self.learning_rate, beta1=0.5)
                d_optimizer = self.d_optimizer
                g_optimizer = self.g_optimizer

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

                    ops = [d_step, g_step]
                    if self.use_encoder and self.encoder=='INDEPENDENT':
                        # If it is not independent it's updated under Discriminator
                        if self.use_tpu:
                            e_optimizer =\
                             tf.contrib.tpu.CrossShardOptimizer(self.e_optimizer)

                        e_step = e_optimizer.minimize(
                            e_loss,
                            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='Encoder'))
                        ops.append(e_step)

                    increment_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
                    ops.append(increment_step)
                    joint_op = tf.group(ops)

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
