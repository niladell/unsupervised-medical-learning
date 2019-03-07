"""Simple generator and discriminator models.

Based on the convolutional and "deconvolutional" models presented in
"Unsupervised Representation Learning with Deep Convolutional Generative
Adversarial Networks" by A. Radford et. al.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
from core import CoreModelTPU

def _leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=0.2)


def _batch_norm(x, is_training, name):
  return tf.layers.batch_normalization(
      x, momentum=0.8, epsilon=1e-5, training=is_training, name=name)


class ToyModel(CoreModelTPU):
    """
    Example definition of a model/network architecture using this template.
    """

    def __init__(self, number_of_gaussians, *kargs, **kwargs):
        self.number_of_gaussians = number_of_gaussians
        super().__init__(*kargs, **kwargs)

    def discriminator(self, x, is_training=True, scope='Discriminator', noise_dim=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            tf.logging.debug('Discriminator %s', self.dataset)
            tf.logging.debug('D -- Input %s', x)

            x = tf.layers.Dense(units=25, name='d_fc_1')(x)
            tf.logging.debug(x)
            x = _leaky_relu(x)
            # x = _batch_norm(x, is_training, name='d_bn1')

            x = tf.layers.Dense(units=10, name='d_fc_2')(x)
            tf.logging.debug(x)
            x = _leaky_relu(x)
            # x = _batch_norm(x, is_training, name='d_bn2')

            # x = tf.layers.Dense(units=50, name='d_fc_3')(x)
            # tf.logging.debug(x)
            # x = _leaky_relu(x)
            # x = _batch_norm(x, is_training, name='d_bn3')

            discriminate = tf.layers.Dense(units=1, name='discriminate')(x)
            tf.logging.debug(discriminate)

            if noise_dim:
                encode = tf.layers.Dense(units=noise_dim, name='encode')(x)
                tf.logging.debug(encode)
                return discriminate, encode

            return discriminate


    def generator(self, x, is_training=True, scope='Generator', reuse=tf.AUTO_REUSE):
        with tf.variable_scope(scope, reuse=reuse):
            tf.logging.debug('Generator %s', self.dataset)
            tf.logging.debug('G -- Input %s', x)

            x = tf.layers.Dense(units=10, name='g_fc_1')(x)
            tf.logging.debug(x)
            x = _leaky_relu(x)
            # x = _batch_norm(x, is_training, name='g_bn1')

            x = tf.layers.Dense(units=25, name='g_fc_2')(x)
            tf.logging.debug(x)
            x = _leaky_relu(x)
            # x = _batch_norm(x, is_training, name='g_bn2')

            # x = tf.layers.Dense(units=500, name='g_fc_3')(x)
            # tf.logging.debug(x)
            # x = _leaky_relu(x)
            # x = _batch_norm(x, is_training, name='g_bn3')

            x = tf.layers.Dense(units=2, name='generated')(x)
            tf.logging.debug(x)

            return x

    def make_config(self):
        """Generates a config file for the tf.Estimators"""

        tpu_cluster_resolver = None
        if self.use_tpu:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                self.tpu,
                zone=self.tpu_zone,
                project=self.gcp_project)

        config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=self.model_dir,
            tpu_config=tf.contrib.tpu.TPUConfig(
                num_shards=self.num_shards,
                iterations_per_loop=self.iterations_per_loop))

        params = {
            'data_dir': self.data_dir,
            'noise_dim': self.noise_dim,
            'noise_cov': self.noise_cov,
            'number_of_gaussians': self.number_of_gaussians
            }
        return config, params



    def sample_from_data(self, generate_input_fn):
        """Sample some images from the data.

        Args:
            input_fn (function): Input function used on the model.

        Returns:
            np.array with the sampled data
        """
        config, params = self.make_config()
        # Get some sample images
        # CPU-based estimator used for PREDICT (generating images)
        data_sampler = tf.contrib.tpu.TPUEstimator(
            model_fn=lambda features, labels, mode, params: tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=features['real_images']),
            use_tpu=False,
            config=config,
            params=params,
            predict_batch_size=self.num_viz_images )

        sample_images = data_sampler.predict(input_fn=generate_input_fn('TRAIN'))
        samples = []
        for i in range(self.num_viz_images ):
            samples.append(next(sample_images))
        samples = np.vstack(samples)
        return samples


    def save_samples_from_data(self, generate_input_fn):
        """Sample some images from the data and print them as images.

        Args:
            input_fn (function): Input function used on the model.
        """
        samples = self.sample_from_data(generate_input_fn)

        plt.figure(figsize=(20,12))
        plt.scatter(samples[:,0], samples[:,1], label='real data')
        plt.legend()
        plt.savefig(self.model_dir + 'real_data' + '.png', bbox_inches='tight')
        plt.close()



    def generate_images(self, generate_input_fn, current_step):
        """     Overwrite generate images function       """

        tf.logging.info('Start generating images')
        # Render some generated images
        rounds_gen_imgs = max(int(np.ceil(self.num_viz_images / self.batch_size)), 1)
        tf.logging.debug('Gonna generate images in %s rounds', rounds_gen_imgs)
        samples = []
        for i in range(rounds_gen_imgs):
            tf.logging.debug('Predict round %s/%s', i, rounds_gen_imgs)
            generated_iter = self.est.predict(input_fn=generate_input_fn('PREDICT'))
            samples += [p['generated_images'] for p in generated_iter]
        samples = np.vstack(samples)

        tf.logging.debug('Genreated %s %s images', len(samples), samples[0].shape)

        r_samples = self.sample_from_data(generate_input_fn)

        plt.figure(figsize=(20,12))
        plt.scatter(r_samples[:,0], r_samples[:,1], label='true dist')
        plt.scatter(samples[:,0], samples[:,1], c='r', label='GAN')
        plt.legend()
        plt.savefig(self.model_dir + '.png',bbox_inches='tight')
        plt.close()

