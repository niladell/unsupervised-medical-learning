""" Core tensorflow model that basically encapsulates all the basic ops
    in order to run an experiment.
"""

import os
import json
import numpy as np
from pprint import pformat

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

from util.image_postprocessing import convert_array_to_image

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
                 use_encoder: bool,
                 encoder: str,
                 e_optimizer: str,
                 e_loss_lambda: float,
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
            use_encoder (bool)
            encoder (str): Which encoder to use. 'ATTACHED' to the discriminator or 'INDEPENDENT' from it.
            e_optimizer (str): Optimizer to use in the encoder. Defaults to ADAM.
            e_loss_lambda (str): Factor by which the encoder loss is scaled (`Loss = Adv_oss + lambda * Enc_loss`)
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
          '{}/{}_{}{}z{}_{}{}{}{}_lr{}'.format(
                    model_dir,
                    self.__class__.__name__,
                    'E' if use_encoder else '',
                    encoder[0] + '_' if use_encoder and encoder else '', # A bit of a stupid option
                    noise_dim,
                    d_optimizer[0],
                    g_optimizer[0],
                    e_optimizer[0] if e_optimizer else '', # TODO a bit of a mess with the encoder options
                    '_ld%s' % e_loss_lambda if use_encoder else '',
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
        self.e_loss_lambda = e_loss_lambda

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
        non_relevant_data = ['use_tpu', 'tpu', 'tpu_zone', 'gcp_project', 'num_shards',
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
                learning_rate=learning_rate)
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


    def create_losses(self, d_logits_real, d_logits_generated, encoded_logits_generated, input_noise):
        """Compute the different losses

        Args:
            d_logits_real: [description]
            d_logits_generated: [description]
            g_logits: [description]
            encoded_logits_generated:
            input_noise:

        Retuns:
            The losses for the Discriminator, Generator, Encoder

        (in case of no encoder the Encoder loss will be none)
        """
        with tf.variable_scope('loss_compute'):
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
            d_loss:
            g_loss:
            e_loss:
        """
        with tf.variable_scope('loss_commbine'):
            with tf.variable_scope('Discriminator'):
                #   Discriminator:
                if self.use_encoder and self.encoder == 'ATTACHED':
                    d_loss = d_loss + self.e_loss_lambda * e_loss
                d_loss = tf.reduce_mean(d_loss)

            with tf.variable_scope('Generator'):
                #   Generator:
                # Do we use the encoder loss to train on G or is it independent
                e_loss_on_g = True
                if self.use_encoder and e_loss_on_g:
                    g_loss = g_loss + self.e_loss_lambda * e_loss
                g_loss = tf.reduce_mean(g_loss)

            with tf.variable_scope('Encoder'):
                #  Encoder:
                if self.use_encoder:
                    e_loss = tf.reduce_mean(e_loss)

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
            d_loss, g_loss, e_loss = self.create_losses(d_on_data_logits, d_on_g_logits, g_logits_encoded, random_noise)

            # Combine losses
            d_loss_train, g_loss_train, e_loss_train = self.combine_losses(d_loss, g_loss, e_loss)

            if mode == tf.estimator.ModeKeys.TRAIN:
                #########
                # TRAIN #
                #########

                # ? TODO is this the best way to deal with the optimziers?
                # d_optimizer = tf.train.GradientDescentOptimizer(
                #     learning_rate=self.learning_rate)
                # d_optimizer = tf.train.AdamOptimizer(
                #     learning_rate=self.learning_rate, beta1=0.5)
                d_optimizer = self.d_optimizer
                g_optimizer = self.g_optimizer
                e_optimizer = self.e_optimizer
                if self.use_tpu:
                    d_optimizer = tf.contrib.tpu.CrossShardOptimizer(d_optimizer)
                    g_optimizer = tf.contrib.tpu.CrossShardOptimizer(g_optimizer)
                    if self.use_encoder and self.encoder=='INDEPENDENT':
                        e_optimizer =\
                             tf.contrib.tpu.CrossShardOptimizer(e_optimizer)

                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    d_step = d_optimizer.minimize(
                        d_loss_train,
                        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='Discriminator'))
                    g_step = g_optimizer.minimize(
                        g_loss_train,
                        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='Generator'))

                    ops = [d_step, g_step]
                    # If it is not independent it's updated under Discriminator
                    if self.use_encoder and self.encoder=='INDEPENDENT':
                        e_step = e_optimizer.minimize(
                            e_loss_train,
                            var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                scope='Encoder'))
                        ops.append(e_step)

                    increment_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
                    ops.append(increment_step)
                    joint_op = tf.group(ops)

                return tf.contrib.tpu.TPUEstimatorSpec(
                        mode=mode,
                        loss=g_loss_train,
                        train_op=joint_op)

            elif mode == tf.estimator.ModeKeys.EVAL:
                ########
                # EVAL #
                ########
                def _eval_metric_fn(d_loss, g_loss, e_loss, d_on_data_logits, d_on_g_logits):
                    #  This thing runs on the same TPU as training so it needs to be
                    # consistent with the TPU restrictions and the shapes imposed in
                    # training
                    with tf.variable_scope('metrics'):
                        d_on_data = tf.sigmoid(d_on_data_logits)
                        d_on_g = tf.sigmoid(d_on_g_logits)

                        accuracy =  1/2*tf.reduce_mean(d_on_data) \
                                + 1/2*tf.reduce_mean(tf.ones_like(d_on_g)-d_on_g)
                        tf.logging.debug(accuracy)
                        metrics = {
                            'discriminator_loss': tf.metrics.mean(d_loss),
                            'generator_loss': tf.metrics.mean(g_loss),
                            'discriminator_accuracy': tf.metrics.mean(accuracy)
                            }
                        if self.use_encoder:
                            metrics['encoder_loss'] = tf.metrics.mean(e_loss)

                        tf.logging.debug('Metrics %s', metrics)

                        return metrics

                if not self.use_encoder:
                    e_loss = None # TODO Quick fix, this is a bit messy, needa refactor

                return tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=tf.reduce_mean(g_loss),
                    eval_metrics=(_eval_metric_fn,
                            [d_loss, g_loss, e_loss, d_on_data_logits, d_on_g_logits]))

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
            'noise_dim': self.noise_dim
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

            self.generate_images(generate_input_fn, current_step)

    def generate_images(self, generate_input_fn, image_name):
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

            # TODO This is a cheap fix, need to change it to a more dynamic thign
            if self.num_viz_images < 100:
                tiled_image = images[0]
            else:
                image_rows = [np.concatenate(images[i:i+10], axis=0)
                                for i in range(0, self.num_viz_images , 10)]
                tiled_image = np.concatenate(image_rows, axis=1)

            img = convert_array_to_image(tiled_image)

            step_string = str(image_name).zfill(6)
            file_obj = tf.gfile.Open(
                os.path.join(self.model_dir,
                                'generated_images', 'gen_%s.png' % (step_string)), 'w')
            img.save(file_obj, format='png')
            tf.logging.info('Finished generating images')

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
        img = convert_array_to_image(tiled_image)

        file_obj = tf.gfile.Open(
            os.path.join(self.model_dir,
                            'generated_images', 'sampled_data.png'), 'w')
        img.save(file_obj, format='png')
        tf.logging.info('File with sample images created.')
