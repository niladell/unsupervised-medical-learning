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


class CoreModelTPU(object):

    def __init__(self,
                 model_dir: str,
                 data_dir: str,
                 dataset: str,
                 learning_rate: float = 0.0002,
                 d_optimizer: str = 'SGD',
                 g_optimizer: str = 'ADAM',
                 noise_dim: int = 64,
                 use_encoder: bool = False,
                 encoder: str = None,
                 e_optimizer: str = None,
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
                 num_shards: int = None
                ):
        """Wrapper class for the model.

        Args:
            model_dir (str): Model directory
            data_dir (str): Data directory
            learning_rate (float, optional): Defaults to 0.0002.
            d_optimizer (str): Optimizer to use in the discriminator. Defaults to SGD.
            g_optimizer (str): Optimizer to use in the generator. Defaults to ADAM.
            noise_dim (int): Size of the nose (or feature) space. Defaults to 64.
            use_encoder (bool): Defaults to False.
            encoder (str): Which encoder to use. 'ATTACHED' to the discriminator or 'INDEPENDENT' from it.
            e_optimizer (str): Optimizer to use in the encoder. Defaults to ADAM.
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
        """
        self.dataset = dataset
        self.data_dir = data_dir
        if model_dir[-1] == '/':
            model_dir = model_dir[:-1]
        self.model_dir =\
          '{}/{}_{}{}z{}_lr{}'.format(
                    model_dir,
                    self.__class__.__name__,
                    'E' if use_encoder else '',
                    encoder[0] + '_' if use_encoder and encoder else '',
                    noise_dim,
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

        self.use_encoder = use_encoder
        if encoder not in ['ATTACHED', 'INDEPENDENT']:
            raise NameError('Encoder type not defined.')
        self.encoder = encoder
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

        if tf.gfile.Exists(self.model_dir):
            if tf.gfile.Exists(self.model_dir + '/params.txt'):
                tf.logging.info('Older params file exists.')
                with tf.gfile.GFile(self.model_dir + '/params.txt', 'rb') as f:
                    old_params = json.loads(f.read())
                if model_params != old_params:
                    tf.logging.info('Is not equal')
                    # TODO Should definitively ignore tpu stuff
                    tf.logging.info('Old parameters:\n{}'.format(pformat(model_params)))
                    raise ValueError('The model in {} should be equal but params.txt differs'.format(self.model_dir))
            else:
                tf.logging.warning('Folder exists but without parameters. ' +\
                  'The code is gonna run assuming the parameters were the ' +\
                  'same (using the ones defined on this session).')
        # print('here')
        # exit()
        with tf.gfile.GFile(self.model_dir + '/params.txt', 'wb') as f:
            f.write(json.dumps(model_params, indent=4, sort_keys=True))

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


    def generate_model_fn(self):
        """Definition of the model function to use. It should return a model_fn
        that takes arguments: features, labels, mode and params; and returns a
        set of TPUEstimatorSpec.

        Do not modify the function here this is a placeholder for the actual
        definition in `model/` (see example)
        """

        raise NotImplementedError('No model function defined')

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

            img = convert_array_to_image(tiled_image)

            step_string = str(current_step).zfill(5)
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
        img = convert_array_to_image(tiled_image)

        file_obj = tf.gfile.Open(
            os.path.join(self.model_dir,
                            'generated_images', 'sampled_data.png'), 'w')
        img.save(file_obj, format='png')
        tf.logging.info('File with sample images created.')
