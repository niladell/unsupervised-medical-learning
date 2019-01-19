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

from util.image_postprocessing import save_array_as_image

tfgan = tf.contrib.gan
queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework


class CoreModelTPU_VAE(object):

    def __init__(self,
                 model_dir: str,
                 data_dir: str,
                 dataset: str,
                 learning_rate: float = 0.0002,
                 optimizer: str = 'SGD',
                 #d_optimizer: str = 'SGD',
                 #g_optimizer: str = 'ADAM',
                 code_dim: int = 64,
                 #use_encoder: bool = False,
                 #encoder: str = None,
                 #e_optimizer: str = None,
                 batch_size: int = 128,
                 iterations_per_loop: int = 100,
                 num_viz_images: int = 100,
                 eval_loss: bool = False,
                 train_steps_per_eval: int = 100,
                 num_eval_images: int = 100,
                 use_tpu: str = False,
                 tpu: str = '',
                 tpu_zone: str = None,
                 gcp_project: str = None,
                 num_shards: int = None,
                 ignore_params_check: bool = False
                ):
        """Wrapper class for the model.

        Args:
            model_dir (str): Model directory
            data_dir (str): Data directory
            learning_rate (float, optional): Defaults to 0.0002.
            optimizer (str): Optimizer to use. Defaults to SGD.
            #d_optimizer (str): Optimizer to use in the discriminator. Defaults to SGD.
            #g_optimizer (str): Optimizer to use in the generator. Defaults to ADAM.
            code_dim (int): Size of the nose (or feature) space. Defaults to 64.
            #use_encoder (bool): Defaults to False.
            #encoder (str): Which encoder to use. 'ATTACHED' to the discriminator or 'INDEPENDENT' from it.
            #e_optimizer (str): Optimizer to use in the encoder. Defaults to ADAM.
            batch_size (int, optional): Defaults to 1024.
            iterations_per_loop (int, optional): Defaults to 500. Iteratios per loop on the estimator.
            num_viz_images (int, optional): Defaults to 100. Number of example images generated.
            eval_loss (bool, optional): Defaults to False.
            train_steps_per_eval (int, optional): Defaults to 100.
            num_eval_images (int, optional): Defaults to 100. Number of eval samples.
            use_tpu (str, optional): Defaults to False.
            tpu (str, optional): Defaults to ''. TPU to use.
            tpu_zone (str, optional): Defaults to None.
            gcp_project (str, optional): Defaults to None.
            num_shards (int, optional): Defaults to None.
            ignore_params_check (bool): Runs without checking parameters form previous runs. Defaults to False.
        """
        self.dataset = dataset
        self.data_dir = data_dir
        self.model_dir = model_dir
        '''
        if model_dir[-1] == '/':
            model_dir = model_dir[:-1]

        self.model_dir =\
          '{}/{}_{}{}z{}_lr{}'.format(
                    model_dir,
                    self.__class__.__name__,
                    'E' if use_encoder else '',
                    encoder[0] + '_' if use_encoder and encoder else '',
                    code_dim,
                    learning_rate)
        '''
        self.use_tpu = use_tpu
        self.tpu = tpu
        self.tpu_zone = tpu_zone
        self.gcp_project = gcp_project
        self.num_shards = num_shards
        self.learning_rate = learning_rate
        self.optimizer = self.get_optimizer(optimizer, learning_rate)
        self.code_dim = code_dim
        self.batch_size = batch_size
        self.iterations_per_loop = iterations_per_loop
        self.num_viz_images = num_viz_images
        self.eval_loss = eval_loss
        self.train_steps_per_eval = train_steps_per_eval
        self.num_eval_images = num_eval_images
        '''
        self.g_optimizer = self.get_optimizer(g_optimizer, learning_rate)
        self.d_optimizer = self.get_optimizer(d_optimizer, learning_rate)
        self.use_encoder = use_encoder
        if encoder not in ['ATTACHED', 'INDEPENDENT']:
            raise NameError('Encoder type not defined.')
        self.encoder = encoder
        self.e_optimizer = None
        if use_encoder:
            self.e_optimizer = self.get_optimizer(e_optimizer, learning_rate)
        '''


        from copy import deepcopy
        model_params = deepcopy(self.__dict__)
        model_params['optimizer'] = optimizer
        #model_params['d_optimizer'] = d_optimizer
        #model_params['g_optimizer'] = g_optimizer
        #model_params['e_optimizer'] = e_optimizer

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
        #with tf.gfile.GFile(self.model_dir + '/params.txt', 'wb') as f:
        #    f.write(json.dumps(model_params, indent=4, sort_keys=True))


    def equal_parms(self, model_params, old_params):
        """Compare the old model parameters with the newly defined ones"""

        # If both are equal
        if model_params == old_params:
            return True, model_params

        # If different this parameters should not affect the model or training outcome
        non_relevant_data = ['use_tpu', 'tpu', 'tpu_zone', 'gcp_project', 'num_shards',
                             'num_viz_images', 'eval_loss', 'train_steps_per_eval',
                             'num_eval_images'] # What else should be here?

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
            model_params[elem] = ' --> '.join([old_params[elem], model_params[elem]])
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

    def discriminator(self, x, is_training=True, scope='Discriminator', code_dim=None): #pylint: disable=E0202
        """
        Definition of the discriminator to use. Do not modify the function here
        placeholder for the actual definition in model/ (see example)

        Args:
            x: Input to the discriminator
            is_training:
            scope: Default Discriminator.
            code_dims: Output size of the encoder (in case there's one)

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
                #random_noise = features['random_noise']
                predictions = {
                    'generated_images': self.generator(
                                            z, is_training=False)}

                return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)

            # Use params['batch_size'] for the batch size inside model_fn
            batch_size = params['batch_size']   # pylint: disable=unused-variable
            code_dim = params['code_dim']
            real_images = features['real_images']
            #random_noise = features['random_noise']

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)
            z_mu, z_log_sigma = tf.squeese(self.discriminator(real_images, code_dim=code_dim))
            z = layers.Lambda(sampling)([z_mu, z_log_sigma])
            generated_images = self.generator(z, is_training=is_training)


            '''
            # Get logits from discriminator

            d_on_data_logits = tf.squeeze(self.discriminator(real_images))
            if self.use_encoder and self.encoder == 'ATTACHED':
                # If we use and embedded encoder we create it here
                d_on_g_logits, g_logits_encoded = self.discriminator(generated_images, code_dim=code_dim)
                d_on_g_logits = tf.squeeze(d_on_g_logits)
            else:
                # Regular GAN w/o encoder
                d_on_g_logits = tf.squeeze(self.discriminator(generated_images))

            z_mean = tf.layers.dense(x, units=n_latent)
            sd= 0.5 * tf.layers.dense(x, units=n_latent)
            epsilon = tf.random_normal(tf.stack([tf.shape(x)[0], n_latent]))
            z  = mn + tf.multiply(epsilon, tf.exp(sd))

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
                    _, g_logits_encoded = self.discriminator(generated_images,
                                                            scope='Encoder',
                                                            code_dim=code_dim)
                e_loss = tf.losses.mean_squared_error(
                    labels=random_noise,
                    predictions=g_logits_encoded)
            '''


            if mode == tf.estimator.ModeKeys.TRAIN:
                #########
                # TRAIN #
                #########
                # TODO There has to be a less messy way of doing theis encoder steps
                #if self.use_encoder and self.encoder == 'ATTACHED':
                #    d_loss = d_loss + e_loss
                #d_loss = tf.reduce_mean(d_loss)
                # Do we use the encoder loss to train on G or is it independent
                #e_loss_on_g = True
                #if self.use_encoder and e_loss_on_g:
                #    g_loss = g_loss + e_loss
                #g_loss = tf.reduce_mean(g_loss)
                # ? TODO is this the best way to deal with the optimziers?
                # d_optimizer = tf.train.GradientDescentOptimizer(
                #     learning_rate=self.learning_rate)
                # d_optimizer = tf.train.AdamOptimizer(
                #     learning_rate=self.learning_rate, beta1=0.5)
                            # Reconstruction loss
                import keras.backend as K
                reconstruction_loss=K.sum(K.binary_crossentropy(generated_images,real_images), axis=1)
                #reconstruction_loss = -tf.reduce_sum(real_images * tf.log(1e-10 + generated_images) \
                #                + (1 - real_images) * tf.log(1e-10 + 1 - generated_images)), 1)
                # KL Divergence loss
                kl_div_loss = -0.5 * tf.reduce_sum(1 + z_std - tf.square(z_mu) - tf.exp(z_log_sigma), 1)

                loss = tf.reduce_mean(reconstruction_loss + kl_div_loss)

                optimizer = self.optimizer
                #d_optimizer = self.d_optimizer
                #g_optimizer = self.g_optimizer

                if self.use_tpu:
                    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
                    #d_optimizer = tf.contrib.tpu.CrossShardOptimizer(d_optimizer)
                    #g_optimizer = tf.contrib.tpu.CrossShardOptimizer(g_optimizer)

                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    step = optimizer.minimize(loss,
                        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='Discriminator')) #????

                    '''
                    d_step = d_optimizer.minimize(
                        d_loss,
                        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='Discriminator'))
                    g_step = g_optimizer.minimize(
                        g_loss,
                        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                    scope='Generator'))
                    '''
                    ops = [d_step, g_step]
                    '''
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
                    '''
                    increment_step = tf.assign_add(tf.train.get_or_create_global_step(), 1)
                    ops.append(increment_step)
                    joint_op = tf.group(ops)

                return tf.contrib.tpu.TPUEstimatorSpec(
                        mode=mode,
                        loss=g_loss,    #??????
                        train_op=joint_op)

            elif mode == tf.estimator.ModeKeys.EVAL:
                ########
                # EVAL #
                ########
                #def _eval_metric_fn(d_loss, g_loss):    #??????
                def _eval_metric_fn(loss):
                # When using TPUs, this function is run on a different machine than the
                # rest of the model_fn and should not capture any Tensors defined there
                    return {
                        'loss': tf.metrics.mean(loss)
                        #'discriminator_loss': tf.metrics.mean(d_loss),
                        #'generator_loss': tf.metrics.mean(g_loss)     #??????
                        }

                return tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=tf.reduce_mean(g_loss),                          #??????
                    eval_metrics=(_eval_metric_fn, [loss]))
                    #eval_metrics=(_eval_metric_fn, [d_loss, g_loss]))    #??????

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
            'code_dim': self.code_dim
            }
        return config, params

    def build_model(self):
        """Builds the tensorflow model"""

        model_fn = self.generate_model_fn()

        tf.logging.info('Start')
        config, params = self.make_config()

        # TPU-based estimator used for TRAIN and EVAL
        self.est = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            use_tpu=self.use_tpu,
            config=config,
            params=params,
            train_batch_size=self.batch_size,
            eval_batch_size=self.batch_size)

        # CPU-based estimator used for PREDICT (generating images)
        self.cpu_est = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            use_tpu=False,
            config=config,
            params=params,
            predict_batch_size=self.num_viz_images)


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

            # Render some generated images
            generated_iter = self.cpu_est.predict(input_fn=generate_input_fn('PREDICT'))
            images = [p['generated_images'][:, :, :] for p in generated_iter]
            if len(images) != self.num_viz_images :
                tf.logging.info('Made %s images (when it should have been %s',
                    len(images), self.num_viz_images )
                images = images[:self.num_viz_images ]

            # assert len(images) == self.num_viz_images
            image_rows = [np.concatenate(images[i:i+10], axis=0)
                            for i in range(0, self.num_viz_images , 10)]
            tiled_image = np.concatenate(image_rows, axis=1)
            step_string = str(current_step).zfill(5)
            filename = os.path.join(self.model_dir,'generated_images', 'gen_%s.png' % (step_string))
            img = save_array_as_image(tiled_image, filename)


            file_obj = tf.gfile.Open(
                os.path.join(self.model_dir,
                                'generated_images', 'gen_%s.png' % (step_string)), 'w')
            img.save(file_obj, format='png')
            tf.logging.info('Finished generating images')



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
        images = []
        for i in range(self.num_viz_images ):
            images.append(next(sample_images))

        image_rows = [np.concatenate(images[i:i+10], axis=0)
                    for i in range(0, self.num_viz_images , 10)]
        tiled_image = np.concatenate(image_rows, axis=1)
        filename = os.path.join(self.model_dir,'generated_images', 'gen_%s_2.png' % (num_viz_images))
        img = save_array_as_image(tiled_image, filename)

        file_obj = tf.gfile.Open(
            os.path.join(self.model_dir,
                            'generated_images', 'sampled_data.png'), 'w')
        img.save(file_obj, format='png')
        tf.logging.info('File with sample images created.')
