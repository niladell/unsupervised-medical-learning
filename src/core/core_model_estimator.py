""" Core tensorflow model that basically encapsulates all the basic ops
    in order to run an experiment.
"""

import os
import json
import numpy as np
from pprint import pformat
import psutil
import gc

import tensorflow as tf
USE_ALTERNATIVE = False
try:
    from tensorflow.data import Dataset
except ImportError:
    tf.logging.warning('Using alternative settings due to old TF version')
    Dataset = tf.data.Dataset
    USE_ALTERNATIVE = True
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver #pylint: disable=E0611
from tensorflow.python.estimator import estimator

from util.image_postprocessing import save_array_as_image, save_windowed_image, slice_windowing
from util.tensorboard_logging import Logger as TFSLogger

tfgan = tf.contrib.gan
queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework

NUMBER_REPLICAS = 8  # TPU Replicas

class CoreModelTPU(object):

    def __init__(self,
                 model_dir: str,
                 data_dir: str,
                 dataset: str,
                 learning_rate: float,
                 d_optimizer: str,
                 g_optimizer: str,
                 noise_dim: int,
                 noise_cov: str,
                 use_wgan_penalty: bool,
                 wgan_lambda: float,
                 wgan_n: int,
                 use_encoder: bool,
                 encoder: str,
                 e_optimizer: str,
                 e_loss_lambda: float,
                 use_window_loss: bool,
                 lambda_window: float,
                 reconstruction_loss: bool,
                 batch_size: int,
                 soft_label_strength: float,
                 iterations_per_loop: int,
                 num_viz_images: int,
                 eval_loss: bool,
                 train_steps_per_eval: int,
                 num_eval_images: int,
                 use_tpu: str,
                 tpu: str,
                 tpu_zone: str,
                 gcp_project: str,
                 num_shards: int,
                 ignore_params_check: bool,
                ):
        """Wrapper class for the model.

        Args:
            model_dir (str): Model directory
            data_dir (str): Data directory
            learning_rate (float): Defaults to 0.0002.
            d_optimizer (str): Optimizer to use in the discriminator. Defaults to SGD.
            g_optimizer (str): Optimizer to use in the generator. Defaults to ADAM.
            noise_dim (int): Size of the nose (or feature) space. Defaults to 64.
            noise_cov (str): Covariance of the random noise to sample. Avail: 'IDENTITY', 'POWER
            use_wgan_penalty (bool)
            wgan_lambda (float): `D_loss = D_loss + lambda * WGAN_penalty`
            wgan_n (int): Number of times that the Discriminator (critic) is updated per step
            use_encoder (bool)
            encoder (str): Which encoder to use. 'ATTACHED' to the discriminator or 'INDEPENDENT' from it.
            e_optimizer (str): Optimizer to use in the encoder. Defaults to ADAM.
            e_loss_lambda (str): Factor by which the encoder loss is scaled (`Loss = Adv_oss + lambda * Enc_loss`)
            window_loss (bool): If to use a second adversarial loss with the window version of the of the data and generated images.
            lambda_window (bool): `Adv. Loss = Regular adv. loss + lambda * Window Adv. loss
            reconstruction_loss (bool): If to compute x->E(x)->z'->G(z')->x' and minimize the loss as if it was a cheap version of a (V)AE
            batch_size (int): Defaults to 1024.
            soft_label_strength (float). Value of the perturbation on soft labels (0 is same as hard labels).
            iterations_per_loop (int): Defaults to 500. Iteratios per loop on the estimator.
            num_viz_images (int): Defaults to 100. Number of example images generated.
            eval_loss (bool)
            train_steps_per_eval (int): Defaults to 100.
            num_eval_images (int): Defaults to 100. Number of eval samples.
            use_tpu (str)
            tpu (str): Defaults to ''. TPU to use.
            tpu_zone (str): Defaults to None.
            gcp_project (str): Defaults to None.
            num_shards (int): Defaults to None.
            ignore_params_check (bool): Runs without checking parameters form previous runs. Defaults to False.
        """
        self.dataset = dataset
        self.data_dir = data_dir
        if model_dir[-1] == '/':
            model_dir = model_dir[:-1]
        self.model_dir =\
          '{}/{}_{}{}{}{}z{}{}_{}{}{}{}{}_lr{}'.format(
                    model_dir,
                    self.__class__.__name__,
                    'r' if reconstruction_loss else '',
                    'E' if use_encoder else '',
                    encoder[0] + '_' if use_encoder and encoder else '', # A bit of a stupid option
                    'Win%s_' % lambda_window if use_window_loss else '',
                    noise_dim,
                    'p' if noise_cov.upper() == 'POWER' else '',
                    d_optimizer[0],
                    g_optimizer[0],
                    e_optimizer[0] if e_optimizer else '', # TODO a bit of a mess with the encoder options
                    '_ld%s' % e_loss_lambda if use_encoder else '',
                    '_W%dL%s' % (wgan_n, wgan_lambda) if use_wgan_penalty else '',
                    learning_rate)

        self.use_tpu = use_tpu
        self.tpu = tpu
        self.tpu_zone = tpu_zone
        self.gcp_project = gcp_project
        self.num_shards = num_shards

        self.learning_rate = learning_rate
        self.g_optimizer = self.get_optimizer(g_optimizer, learning_rate)
        self.d_optimizer = self.get_optimizer(d_optimizer, learning_rate)
        self.noise_dim = noise_dim
        self.noise_cov = noise_cov
        self.e_loss_lambda = e_loss_lambda

        self.use_window_loss = use_window_loss
        self.lambda_window = lambda_window

        self.reconstruction_loss = reconstruction_loss

        self.wgan_penalty = use_wgan_penalty
        self.wgan_lambda = wgan_lambda
        self.wgan_n = wgan_n # Number of times that the Discriminator (critic) is updated per step
        self.soft_label_strength = soft_label_strength

        self.use_encoder = use_encoder
        self.encoder = encoder.upper()
        if use_encoder and self.encoder not in ['ATTACHED', 'INDEPENDENT']:
            raise NameError('Encoder type not defined.')
        self.e_optimizer = None
        if use_encoder:
            self.e_optimizer = self.get_optimizer(e_optimizer, learning_rate)

        self.batch_size = batch_size
        self.iterations_per_loop = iterations_per_loop

        self.num_viz_images = num_viz_images
        self.eval_loss = eval_loss
        self.train_steps_per_eval = train_steps_per_eval
        self.num_eval_images = num_eval_images

        from copy import deepcopy
        model_params = deepcopy(self.__dict__)
        model_params['d_optimizer'] = d_optimizer
        model_params['g_optimizer'] = g_optimizer
        model_params['e_optimizer'] = e_optimizer

        tf.logging.info('Current parameters: {}'.format(pformat(model_params)))

        if not tf.gfile.Exists(self.model_dir):
            # Redundant but may we need this to do dry runs?
            tf.gfile.MakeDirs(self.model_dir)

        if ignore_params_check:
            tf.logging.warning('--ignore_params_check is set to True. The model is ' +\
                'not gonna check for compatibility with the previous parameters and ' +\
                'will overwrite params.txt file if it existed already.')
        else:
            if tf.gfile.Exists(self.model_dir):
                if tf.gfile.Exists(self.model_dir + '/params.txt'):
                    tf.logging.info('Older params file exists.')
                    with tf.gfile.GFile(self.model_dir + '/params.txt', 'rb') as f:
                        old_params = json.loads(f.read())
                    equal, model_params = self.equal_parms(model_params, old_params)
                    if not equal:
                        raise ValueError('The following parameters in params.txt differ: \n{}'.format(pformat(model_params)))
                else:
                    tf.logging.warning('Folder exists but without parameters. ' +\
                    'The code is gonna run assuming the parameters were the ' +\
                    'same (but using the ones defined on this session).')

            # Save the params (or the updated version with unrelevant changes)
            with tf.gfile.GFile(self.model_dir + '/params.txt', 'wb') as f:
                f.write(json.dumps(model_params, indent=4, sort_keys=True))


    def equal_parms(self, model_params, old_params):
        """Compare the old model parameters with the newly defined ones"""

        # If both are equal
        if model_params == old_params:
            return True, model_params

        # If different this parameters should not affect the model or training outcome
        non_relevant_data = ['data_dir', 'use_tpu', 'tpu', 'tpu_zone', 'gcp_project', 'num_shards',
                             'num_viz_images', 'eval_loss', 'train_steps_per_eval',
                             'num_eval_images', 'batch_size', 'iterations_per_loop'] # What else should be here?

        def compare(old, new):
            old_keys = set(old.keys())
            new_keys = set(new.keys())
            intersect_keys = old_keys.intersection(new_keys)
            if len(intersect_keys) != len(old_keys):
                raise ValueError('The model parameters have different elements (options). ' +\
                        'If this was expected run the model with the --ignore_param_check')
            modified = {o : (old[o], new[o]) for o in intersect_keys if old[o] != new[o]}
            return modified
        modified = compare(old_params, model_params)

        # If the changes are gonna be critical to the model
        relevant_changes = {}
        for elem in modified:
            if elem not in non_relevant_data:
                relevant_changes[elem] = ' OLD: {} --> NEW: {}'.format(old_params[elem], model_params[elem])
        if relevant_changes:
            return False, relevant_changes

        # If there are changes but should not affect teh model
        tf.logging.warning('There have been parameter changes but these are unrelated to the model')
        for elem in modified:
            model_params[elem] = ' --> '.join([str(old_params[elem]), str(model_params[elem])])
        tf.logging.warning(pformat(model_params))
        return True, model_params

    def get_optimizer(self, name, learning_rate):
        """Create an optimizer

        Available names: 'ADAM', 'SGD'
        """
        # TODO change learning rate for a dict with all the posible \
        # parameters e.g. ADAM: learning rate, epsilon, beta1, beta2..
        if name == 'ADAM':
            return tf.train.AdamOptimizer(
                learning_rate=learning_rate, beta1=0.0, beta2=0.9, epsilon=1e-4)
        elif name == 'SGD':
            return tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        else:
            raise NameError('Optimizer {} not recognised'.format(name))

    def discriminator(self, x, is_training=True, scope='Discriminator', noise_dim=None, reuse=tf.AUTO_REUSE): #pylint: disable=E0202
        """
        Definition of the discriminator to use. Do not modify the function here
        placeholder for the actual definition in model/ (see example)

        Args:
            x: Input to the discriminator
            is_training:
            scope: Default Discriminator.
            noise_dims: Output size of the encoder (in case there's one)
            reuse: Default tf.AUTO_REUSE. Reuse in tf.varaible_scope; just
                modify in the case of wanting an independent network using
                the discriminator architecture (as in an Indep. Encoder)

        Raises:
            NotImplementedError: Model has to be implemented yet (in a separate instance in model/)
        """

        raise NotImplementedError('No discriminator defined.')

    def generator(self, x, is_training=True, scope='Generator'):
        """
        Definition of the generator to use. Do not modify the function here
        placeholder for the actual definition in model/ (see example)

        Args:
            x: Input to the generator
            is_training:
            scope: Default Discriminator.

        Raises:
            NotImplementedError: Model has to be implemented yet (in a separate instance in model/)
        """

        raise NotImplementedError('No discriminator defined.')


    def create_losses(self, d_logits_real, d_logits_generated, encoded_logits_generated, input_noise, real_images=None, generated_images=None, batch_size=None):
        """Compute the different losses

        Args:
            d_logits_real: Logits from the discriminator on real data
            d_logits_generated: Logits from the discriminator on generated
                (fake) data.
            encoded_logits_generated: Logits from the encoder on the generated
                data.
            input_noise: Random noise that was used as input in the generator.
            real_images: Needed for the WGAN loss
            generated_images: Needed for the WGAN loss

        Retuns:
            The losses for the Discriminator, Generator, Encoder

        (in case of no encoder the Encoder loss will be none)
        """
        with tf.variable_scope('loss_compute'):
            if self.wgan_penalty:
                assert real_images is not None and generated_images is not None
                with tf.variable_scope('wgan_loss'):
                    # TODO Not working, WGAN value explodes
                    # WGAN uses a critic intead of a discriminator (i.e. at the end of the day we are
                    # not interested on the real-fake dilema, but on how similar is the fake one to a
                    # real one (wavy interpretation))

                    # WE don't reduce_mean here as we need to pass a tensor with the batch still present
                    # in the EVAL mode. The reduction is done later, just before training.
                    # d_loss =  tf.reduce_mean(d_logits_generated) - tf.reduce_mean(d_logits_real)
                    # g_loss = -tf.reduce_mean(d_logits_generated)
                    d_loss =  d_logits_generated - d_logits_real
                    g_loss = -d_logits_generated


                    tf.logging.debug('WGAN -- D loss: %s', d_loss)
                    tf.logging.debug('WGAN -- G loss: %s', g_loss)

                    difference = real_images - generated_images
                    eps_shape = [batch_size] + [1] * (difference.shape.ndims - 1)
                    eps = tf.random_uniform(shape=eps_shape)
                    x_hat = generated_images + (eps * difference)
                    # with tf.name_scope(None):
                    #     with tf.variable_scope('Discriminator','gpenalty_dscope',
                    #                             reuse=tf.AUTO_REUSE):
                    d_hat = self.discriminator(x_hat)
                    tf.logging.debug('WGAN -- Eps: %s', eps)
                    tf.logging.debug('WGAN -- x^ : %s', x_hat)
                    tf.logging.debug('WGAN -- D^ : %s', d_hat)
                    gradients = tf.gradients(d_hat, x_hat)[0]
                    tf.logging.debug('WGAN -- Grad_x(D): %s', gradients)
                    wgan_penalty = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1,2,3]) + 1e10)
                    wgan_penalty = tf.square(wgan_penalty - 1)

                    tf.logging.debug('WGAN -- Penalty: %s', wgan_penalty)
                    d_loss = d_loss + self.wgan_lambda*wgan_penalty
                    tf.logging.debug('WGAN -- Loss: %s', d_loss)

                    # I am experiencing problems with exploding loss. My main candidate is the
                    # euclidian norm on the gradients (at the end of the day it is a 512*512 tensor)
                    # so it'd be normal for that value to be hughe.
                    # For that reason. Now I'm clipping the maximum loss (which I doubt should be
                    # bigger than 1000)
                    d_loss = tf.clip_by_value(d_loss, -1000, 1000)
                    g_loss = tf.clip_by_value(g_loss, -1000, 1000)


            else:
                with tf.variable_scope('classicGAN_loss'):
                    # Create the labels
                    true_label = tf.ones_like(d_logits_real)
                    fake_label = tf.zeros_like(d_logits_generated)
                    #  We invert the labels for the generator training (ganTricks)
                    true_label_g = tf.ones_like(d_logits_generated)

                    # Soften the labels (ganTricks)
                    if self.soft_label_strength != 0:
                        true_label += tf.random_uniform(true_label.shape,
                                                    minval=-self.soft_label_strength,
                                                    maxval=self.soft_label_strength)
                        true_label = tf.clip_by_value(
                                                true_label, 0, 1)

                        fake_label += tf.random_uniform(fake_label.shape,
                                                    minval=-self.soft_label_strength,
                                                    maxval=self.soft_label_strength)
                        fake_label = tf.clip_by_value(
                                                fake_label, 0, 1)

                        true_label_g += tf.random_uniform(true_label_g.shape,
                                                    minval=-self.soft_label_strength,
                                                    maxval=self.soft_label_strength)
                        true_label_g = tf.clip_by_value(
                                                true_label_g, 0, 1)
                    with tf.variable_scope('Discriminator'):
                        # Calculate discriminator loss
                        d_loss_on_data = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=true_label,
                            logits=d_logits_real)
                        d_loss_on_gen = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=fake_label,
                            logits=d_logits_generated)

                        d_loss = d_loss_on_data + d_loss_on_gen

                    with tf.variable_scope('Generator'):
                        # Calculate generator loss
                        g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=true_label_g,
                            logits=d_logits_generated)

            with tf.variable_scope('Encoder'):
                # Create independent encoder loss
                if self.use_encoder:
                    e_loss = tf.losses.mean_squared_error(
                        labels=input_noise,
                        predictions=encoded_logits_generated,
                        reduction=tf.losses.Reduction.NONE)
                    e_loss = tf.reduce_mean(e_loss, axis=1)
                    tf.logging.debug('Input noise: %s; Encoded logits %s',
                        input_noise, encoded_logits_generated)
                else:
                    e_loss = None

            tf.logging.debug('D loss: %s', d_loss)
            tf.logging.debug('G loss: %s', g_loss)
            tf.logging.debug('E loss: %s', e_loss)

            return d_loss, g_loss, e_loss

    def combine_losses(self, d_loss, g_loss, e_loss):
        """ Combine the losses and return the final loss that is gonna be used on training
        Args:
            d_loss: Discriminator (or critic) loss
            g_loss: Generator loss
            e_loss: Encoder loss

        Retruns:
            The losses for the Discriminator, Generator, Encoder
        """
        if e_loss is not None:
            with tf.variable_scope('loss_commbine'):
                with tf.variable_scope('Discriminator'):
                    #   Discriminator:
                    if self.use_encoder and self.encoder == 'ATTACHED':
                        d_loss = d_loss + self.e_loss_lambda * e_loss

                with tf.variable_scope('Generator'):
                    #   Generator:
                    # Do we use the encoder loss to train on G or is it independent
                    e_loss_on_g = True
                    if self.use_encoder and e_loss_on_g:
                        g_loss = g_loss + self.e_loss_lambda * e_loss

                with tf.variable_scope('Encoder'):
                    #  Encoder:
                    if self.use_encoder:
                        e_loss = tf.reduce_mean(e_loss)

        d_loss = tf.reduce_mean(d_loss)
        g_loss = tf.reduce_mean(g_loss)

        tf.logging.debug('training D loss: %s',d_loss)
        tf.logging.debug('training G loss: %s',g_loss)
        tf.logging.debug('training E loss: %s',e_loss)

        return d_loss, g_loss, e_loss

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
            noise_dim = params['noise_dim']
            real_images = features['real_images']
            random_noise = features['random_noise']

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            generated_images = self.generator(random_noise,
                                              is_training=is_training)

            # Get logits from discriminator for real data
            d_on_data_logits = tf.squeeze(self.discriminator(real_images))

            # Get logits from the encoder and discriminator for the generator
            if self.use_encoder:
                if self.encoder == 'ATTACHED':
                     # If we use and embedded encoder we create alongside the discriminator
                    d_on_g_logits, g_logits_encoded =\
                       self.discriminator(generated_images, noise_dim=noise_dim)
                elif self.encoder == 'INDEPENDENT':
                    _, g_logits_encoded = self.discriminator(generated_images,
                                                            scope='Encoder',
                                                            noise_dim=noise_dim,
                                                            reuse=None)
                    d_on_g_logits = tf.squeeze(self.discriminator(generated_images,
                                                            scope='Discriminator'))
                else:
                    raise NameError('No encoder type {} defined'.format(self.encoder))
            else:
                # Regular GAN w/o encoder
                g_logits_encoded = None
                d_on_g_logits = self.discriminator(generated_images)
            d_on_g_logits = tf.squeeze(d_on_g_logits)


            tf.logging.debug('D real logits %s', d_on_g_logits)
            tf.logging.debug('D fake (G) logits %s', d_on_data_logits)
            tf.logging.debug('E (from G) logits %s', g_logits_encoded)

            # Compute the losses
            d_loss, g_loss, e_loss = self.create_losses(d_logits_real=d_on_data_logits,
                                                        d_logits_generated=d_on_g_logits,
                                                        encoded_logits_generated=g_logits_encoded,
                                                        input_noise=random_noise,
                                                        real_images=real_images,
                                                        generated_images=generated_images,
                                                        batch_size=batch_size)

            if self.use_window_loss:
                with tf.variable_scope('Window'):
                    # TODO is this the optimal brain window?
                    window = (-0.3, -0.2)
                    # perturb_down = 0.1 * np.random.random() - 0.05
                    # perturb_up = 0.1 * np.random.random() - 0.05
                    # windwo = (perturb_down + window[0], perturb_up + window[1])
                    real_window = slice_windowing(real_images, window=window, up_val=1, low_val=0)
                    generated_window = slice_windowing(real_images, window=window, up_val=1, low_val=0)
                    d_on_real_window = tf.squeeze(self.discriminator(real_window))
                    d_on_g_window = tf.squeeze(self.discriminator(generated_window))
                    d_loss_window, g_loss_window, _ = self.create_losses(d_logits_real=d_on_data_logits,
                                                                        d_logits_generated=d_on_g_logits,
                                                                        encoded_logits_generated=g_logits_encoded,
                                                                        input_noise=random_noise,
                                                                        real_images=real_images,
                                                                        generated_images=generated_images,
                                                                        batch_size=batch_size)
                    d_loss = d_loss + self.lambda_window * d_loss_window
                    g_loss = g_loss + self.lambda_window * g_loss_window

            if self.reconstruction_loss:
                if not self.use_encoder or self.use_encoder and self.encoder.upper() == 'INDEPENDENT':
                    raise NotImplementedError('Reconstruction loss not implemented for Independent encoder')
                    # TODO Implement for independent
                with tf.variable_scope('reconstruction'):
                    _, projected_img = self.discriminator(real_images, noise_dim=noise_dim)
                    reconstructed_img = self.generator(projected_img)

                    r_loss = tf.losses.mean_squared_error(
                        labels=tf.layers.flatten(real_images),
                        predictions=tf.layers.flatten(reconstructed_img),
                        reduction=tf.losses.Reduction.NONE)
                    tf.logging.debug('R Loss %s', r_loss)
                    r_loss = tf.reduce_sum(r_loss, axis=1)
                    tf.logging.debug('R Loss %s', r_loss)
                    r_loss_train = tf.reduce_mean(r_loss)
                    tf.logging.debug('R Loss train %s', r_loss_train)
            # Combine losses
            d_loss_train, g_loss_train, e_loss_train = self.combine_losses(d_loss, g_loss, e_loss)
            # d_loss_train, g_loss_train, e_loss_train = d_loss, g_loss, e_loss


            if mode == tf.estimator.ModeKeys.TRAIN:
                ####################################
                #              TRAIN              #
                ###################################

                d_optimizer = self.d_optimizer
                g_optimizer = self.g_optimizer
                e_optimizer = self.e_optimizer
                r_optimizer = self.get_optimizer('ADAM', self.learning_rate) # TODO Badly hardcoded
                if self.use_tpu:
                    d_optimizer = tf.contrib.tpu.CrossShardOptimizer(d_optimizer)
                    g_optimizer = tf.contrib.tpu.CrossShardOptimizer(g_optimizer)
                    if self.use_encoder and self.encoder=='INDEPENDENT':
                        e_optimizer =\
                             tf.contrib.tpu.CrossShardOptimizer(e_optimizer)
                        r_optimizer =\
                             tf.contrib.tpu.CrossShardOptimizer(r_optimizer)

                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    critic_steps = self.wgan_n if self.wgan_penalty else 1
                    ops = []
                    for i in range(critic_steps):
                        d_step = d_optimizer.minimize(
                            d_loss_train,
                            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                        scope='Discriminator'))
                        # I really have no clue if this is a proper replacement for calling
                        # the optimizer op 'n' times, I can't help to feel this is somewhat
                        # weird to begin with, tho it somewhat worked...
                        ops.append(d_step)


                    g_step = g_optimizer.minimize(
                        g_loss_train,
                        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='Generator'))

                    ops.append(g_step)
                    # If it is not independent it's updated under Discriminator
                    if self.use_encoder and self.encoder=='INDEPENDENT':
                        e_step = e_optimizer.minimize(
                            e_loss_train,
                            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='Encoder'))
                        ops.append(e_step)

                    if self.reconstruction_loss:
                        r_step = r_optimizer.minimize(
                            r_loss_train)
                            # var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                            #                     scope='Encoder')) # FOR ALL?
                        ops.append(r_step)


                    increment_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
                    ops.append(increment_step)
                    joint_op = tf.group(ops)

                    tf.logging.debug('Train OPS %s', ops)
                    tf.logging.debug('Joint OP  %s', joint_op)

                return tf.contrib.tpu.TPUEstimatorSpec(
                        mode=mode,
                        loss=g_loss_train,
                        train_op=joint_op)

            elif mode == tf.estimator.ModeKeys.EVAL:
                ########
                # EVAL #
                ########
                def _eval_metric_fn(d_loss, g_loss, e_loss, r_loss, d_on_data_logits, d_on_g_logits):
                    #  This thing runs on the same TPU as training so it needs to be
                    # consistent with the TPU restrictions and the shapes imposed in
                    # training
                    tf.logging.debug('Eval_fn')
                    with tf.variable_scope('metrics'):
                        d_on_data = tf.sigmoid(d_on_data_logits)
                        d_on_g = tf.sigmoid(d_on_g_logits)

                        accuracy = 1/2*tf.reduce_mean(d_on_data) \
                                 + 1/2*tf.reduce_mean(tf.ones_like(d_on_g)-d_on_g)
                        tf.logging.debug(accuracy)
                        metrics = {
                            'discriminator_loss': tf.metrics.mean(d_loss),
                            'generator_loss': tf.metrics.mean(g_loss),
                            'discriminator_accuracy': tf.metrics.mean(accuracy)
                            }
                        if self.use_encoder:
                            metrics['encoder_loss'] = tf.metrics.mean(e_loss)
                        if self.reconstruction_loss:
                            metrics['reconstruction_loss']: tf.metrics.mean(r_loss)

                        tf.logging.debug('Metrics %s', metrics)

                        return metrics

                tf.logging.debug('Start eval')
                if not self.use_encoder:
                    e_loss = tf.zeros_like(g_loss) # TODO Quick fix, this is a bit messy, needa refactor
                    tf.logging.debug('Not using e_loss eval')

                if not self.reconstruction_loss:
                    r_loss = tf.zeros_like(g_loss) # TODO Quick fix, this is a bit messy, needa refactor
                    tf.logging.debug('Not using r_loss eval')
                tf.logging.debug('Inputs to eval %s %s %s %s %s %s', d_loss, g_loss, e_loss, r_loss, d_on_data_logits, d_on_g_logits)
                return tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=tf.reduce_mean(g_loss),
                    eval_metrics=(_eval_metric_fn,
                            [d_loss, g_loss, e_loss, r_loss, d_on_data_logits, d_on_g_logits]))

            # Should never reach here
            raise ValueError('Invalid mode provided to model_fn')
        return model_fn

    def make_config(self):
        """Generates a config file for the tf.Estimators"""

        tpu_cluster_resolver = None
        if self.use_tpu:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                self.tpu,
                zone=self.tpu_zone,
                project=self.gcp_project)

        config = tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=self.model_dir,
            tpu_config=tf.contrib.tpu.TPUConfig(
                num_shards=self.num_shards,
                iterations_per_loop=self.iterations_per_loop))

        params = {
            'data_dir': self.data_dir,
            'noise_dim': self.noise_dim,
            'noise_cov': self.noise_cov
            }
        return config, params

    def build_model(self):
        """Builds the tensorflow model"""
        tf.logging.info('Start')

        model_fn = self.generate_model_fn()
        config, params = self.make_config()

        # Batch needs to be multiple of number of replicas
        mod_num_viz_imgages = self.num_viz_images % NUMBER_REPLICAS
        pred_batch = self.num_viz_images + mod_num_viz_imgages
        pred_batch = min(self.batch_size, pred_batch)

        # TPU-based estimator used for TRAIN, EVAL and PREDICT
        self.est = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            use_tpu=self.use_tpu,
            config=config,
            params=params,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size,
            predict_batch_size=pred_batch
        )

    def train(self,
              train_steps,
              generate_input_fn):
        """Train the model

        Args:
            train_steps (int): Numer of training steps
            generate_input_fn (function): Function that resturns input_fn
            function. (see example or tf.Estimator documentation)
            noise_input_fn (function): input_fn that returns a noise vector
        """

        current_step = estimator._load_global_step_from_checkpoint_dir(self.model_dir)   # pylint: disable=protected-access,line-too-long
        tf.logging.info('Starting training for %d steps, current step: %d' %
                        (train_steps, current_step))
        tf.gfile.MakeDirs(os.path.join(self.model_dir, 'generated_images'))

        # self.generate_images(generate_input_fn, current_step)

        while current_step < train_steps:
            next_checkpoint = int(min(current_step + self.train_steps_per_eval,
                                    train_steps))
            tf.logging.info('Step: %s  -- (Next checkpoint %s)', current_step, next_checkpoint)
            self.est.train(input_fn=generate_input_fn('TRAIN'),
                        max_steps=next_checkpoint)
            current_step = next_checkpoint
            tf.logging.info('Finished training step %d' % current_step)

            if self.eval_loss:
                # Evaluate loss on test set
                metrics = self.est.evaluate(input_fn=generate_input_fn('EVAL'),
                                        steps=max(self.num_eval_images // self.batch_size,1))
                tf.logging.info('Finished evaluating')
                tf.logging.info(metrics)

            # self.generate_images(generate_input_fn, current_step)
            gc.collect()  # I'm experiencing some kind of memory leak (and seems that other people
                          # did too). So, seeing that adding 52GBs of RAM doesn't help I'll just
                          # take the garbage out manually for now.

    def generate_images(self, generate_input_fn, current_step):
        tf.logging.info('Start generating images')
        # Render some generated images
        rounds_gen_imgs = max(int(np.ceil(self.num_viz_images / self.batch_size)), 1)
        tf.logging.debug('Gonna generate images in %s rounds', rounds_gen_imgs)
        images = []
        for i in range(rounds_gen_imgs):
            tf.logging.debug('Predict round %s/%s', i, rounds_gen_imgs)
            generated_iter = self.est.predict(input_fn=generate_input_fn('PREDICT'))
            images += [p['generated_images'][:, :, :] for p in generated_iter]

        if len(images) != self.num_viz_images :
            tf.logging.warning('Made %s images (when it should have been %s)',
                len(images), self.num_viz_images )
            images = images[:self.num_viz_images ]
        tf.logging.debug('Genreated %s %s images', len(images), images[0].shape)

        try:

            custom_tfs = TFSLogger(log_dir=self.model_dir)

            step_string = str(current_step).zfill(6)
            n_log_imgs = 15 if 15 < len(images) else len(images)
            custom_tfs.log_images(
                            tag='individial/gen_%s' % step_string,
                            images=images[:n_log_imgs],
                            step=current_step)
            for idx, img in enumerate(images[:n_log_imgs]):
                custom_tfs.log_histogram(
                                tag='gen_hist_img%s' % idx,
                                values=img,
                                step=current_step,
                                bins=100)
            tf.logging.info('Images and histogram saved to the tf.summary via ' +\
                            'custom TF Summary Logger')

            # TODO This is a cheap fix, need to change it to a more dynamic thign
            if self.num_viz_images < 100:
                tiled_image = images[0]
            else:
                image_rows = [np.concatenate(images[i:i+10], axis=0)
                                for i in range(0, self.num_viz_images , 10)]
                tiled_image = np.concatenate(image_rows, axis=1)

                custom_tfs = TFSLogger(log_dir=self.model_dir)
                custom_tfs.log_images(
                            tag='packed/gen_%s' % step_string,
                            images=[tiled_image],
                            step=current_step)

            filename = os.path.join(self.model_dir,
                                'generated_images', 'gen_%s.png' % (step_string))
            save_array_as_image(tiled_image, filename)

            filename = os.path.join(self.model_dir,
                                'generated_images', 'gen_%s_brainWin.png' % (step_string))
            save_windowed_image(tiled_image, filename)

            tf.logging.info('Finished generating images')

        except MemoryError as e:
            tf.logging.error('%s, trying to save a single image')
            tf.logging.error('Memory usage at {}'.format(psutil.virtual_memory()))
            try:
                step_string = str(current_step).zfill(6)
                filename = os.path.join(self.model_dir,
                                    'generated_images', 'gen_%s_bkup.png' % (step_string))
                save_array_as_image(images[0], filename)
                tf.logging.info('Single image saved!')
            except MemoryError as e:
                tf.logging.error('%s: NO IMAGES WERE GENERATED', e)

    def set_up_encoder(self, batch_size):
        """ Creates the TF Estimator for the encoder predictions.

        Args:
            batch_size (int)
        """

        def encode_fn(features, labels, mode, params):
            del labels    # Unconditional GAN does not use labels
            # Pass images to be encoded
            image = features
            _, encoded_image = self.discriminator(
                                        image, is_training=False, noise_dim=self.noise_dim)
            return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=encoded_image)

        config, params = self.make_config()

        # CPU-based estimator used for ECODE PREDICT (encoding images)
        self.encode_est = tf.contrib.tpu.TPUEstimator(
            model_fn=encode_fn,
            use_tpu=False,
            config=config,
            params=params,
            predict_batch_size=batch_size)


    def encode(self, images, batch_size=1, clean_encoder_est=False):
        """ Encode an image to the Z-space using the model encoder

        Args:
            images (np.array): Image(s) to encode use formats "NHWC",
                "HWC" (single image) or "HW" (single image)
            batch_size (int, optional): Defaults to 1.
            clean_encoder_est (bool, optional): Defaults to False. Resets the
                encoder Estimator at the begining of this function and deletes
                it at the end of it.

        Returns:
            Encoded images
        """
        # If the input is a single image we create a dimension for the batch
        if len(images.shape) < 4:
            images = np.expand_dims(images, axis=0)
        # If the input was a 2D matrix we expand to a single channel
        if len(images.shape) < 4:
            images = np.expand_dims(images, axis=3)
        assert len(images.shape) == 4

        # TODO I don't know if cheking the first image is enough
        if type(images[0,0,0,0]) == np.uint8:
            images = (2 * (images / 255.0) - 1).astype(np.float32)

        if not hasattr(self, 'encode_est') or clean_encoder_est:
            self.set_up_encoder(batch_size)
        def input_fn(params):
            del params
            dataset = tf.data.Dataset.from_tensor_slices((images, [[]]))
            dataset = dataset.batch(batch_size)
            features, labels = dataset.make_one_shot_iterator().get_next()
            return features, labels

        encoded_images = [z for z in self.encode_est.predict(input_fn=input_fn)]
        if clean_encoder_est: #? May we need to free up some memory?
            del self.encode_est
        return encoded_images


    def save_samples_from_data(self, generate_input_fn):
        """Sample some images from the data and print them as images.

        Args:
            input_fn (function): Input function used on the model.
        """

        tf.gfile.MakeDirs(os.path.join(self.model_dir, 'generated_images'))

        config, params = self.make_config()
        # Get some sample images
        # CPU-based estimator used for PREDICT (generating images)
        data_sampler = tf.contrib.tpu.TPUEstimator(
            model_fn=lambda features, labels, mode, params: tpu.TPUEstimatorSpec(mode=mode, predictions=features['real_images']),
            use_tpu=False,
            config=config,
            params=params,
            predict_batch_size=self.num_viz_images )

        sample_images = data_sampler.predict(input_fn=generate_input_fn('TRAIN'))
        tf.logging.info('That ran')
        if self.num_viz_images < 100:
            tiled_image = next(sample_images)
        else:
            images = []
            for i in range(self.num_viz_images ):
                images.append(next(sample_images))

            image_rows = [np.concatenate(images[i:i+10], axis=0)
                        for i in range(0, self.num_viz_images , 10)]
            tiled_image = np.concatenate(image_rows, axis=1)

        filename = os.path.join(self.model_dir,
                            'generated_images', 'sampled_data.png')
        save_array_as_image(tiled_image, filename)

        filename = os.path.join(self.model_dir,
                            'generated_images', 'sampled_data_brainWin.png')
        save_windowed_image(tiled_image, filename)

        tf.logging.info('File with sample images created.')
