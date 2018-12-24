import sys, os

from absl import flags
from absl import logging
# import coloredlogs
logging.set_verbosity(logging.INFO)
#coloredlogs.install(level='INFO')
from os import walk
import pickle
import numpy as np
import functools

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver #pylint: disable=E0611
from tensorflow.python.estimator import estimator

tfgan = tf.contrib.gan
queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework

# from core import DataManager
# from model import ExampleModel
# from datamanager import CIFAR10

SAMPLE_NUM = 50000
CHANNELS = 3
SIDE = 32

BATCH_SIZE = 128
NOISE_DIMS = 64
_NUM_VIZ_IMAGES = 100   # For generating a 10x10 grid of generator samples


tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = flags.FLAGS


# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default='node-1',
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
flags.DEFINE_string(
    'gcp_project', default=None,
    help='Project name for the Cloud TPU-enabled project. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')
flags.DEFINE_string(
    'tpu_zone', default=None,
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Model specific paramenters
flags.DEFINE_string('dataset', 'mnist',
                    'One of ["mnist", "cifar"]. Requires additional flags')
flags.DEFINE_string('model_dir', '', 'Output model directory')
flags.DEFINE_integer('noise_dim', 64,
                     'Number of dimensions for the noise vector')
flags.DEFINE_integer('batch_size', 1024,
                     'Batch size for both generator and discriminator')
flags.DEFINE_integer('num_shards', None, 'Number of TPU chips')
flags.DEFINE_integer('train_steps', 10000, 'Number of training steps')
flags.DEFINE_integer('train_steps_per_eval', 1000,
                     'Steps per eval and image generation')
flags.DEFINE_integer('iterations_per_loop', 100,
                     'Steps per interior TPU loop. Should be less than'
                     ' --train_steps_per_eval')
flags.DEFINE_float('learning_rate', 0.0002, 'LR for both D and G')
flags.DEFINE_boolean('eval_loss', False,
                     'Evaluate discriminator and generator loss during eval')
flags.DEFINE_boolean('use_tpu', True, 'Use TPU for training')


FLAGS(sys.argv)
### HIGHLY PROTOTIPY VERSON BELOW -- TODO MOVE TO DATA MANAGER

def input_fn():
    """Read CIFAR input data from a TFRecord dataset.
    
    Function taken from tensorflow/tpu cifar_keras repo"""
    # del params
    batch_size = BATCH_SIZE
    def parser(serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                "image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features["image"], tf.uint8)
        image.set_shape([3*32*32])
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image = tf.reshape(image, [32, 32, 3])
        label = tf.cast(features["label"], tf.int32)
        logging.info(image)
        noise = tf.random_normal([NOISE_DIMS])
        return {'noise': noise, 'images': image} #, label

    # TEMPORAL
    image_files = ['gs://iowa_bucket/cifar-10-data/train.tfrecords']

    dataset = tf.data.TFRecordDataset([image_files])
    dataset = dataset.map(parser, num_parallel_calls=batch_size)
    dataset = dataset.prefetch(4 * batch_size).cache().repeat()
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)
    return dataset

def generate_input_fn(is_training):
  """Creates input_fn depending on whether the code is training or not."""
  return input_fn()

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)

def generator_fn(noise, weight_decay=2.5e-5, is_training=True):
    """Simple generator to produce MNIST images.

    Args:
        noise: A single Tensor representing noise.
        weight_decay: The value of the l2 weight decay.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population 
            statistics.

    Returns:
        A generated image in the range [-1, 1].
    """
    with framework.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)),\
    framework.arg_scope([layers.batch_norm], is_training=is_training,
                        zero_debias_moving_mean=True):
        logging.info(noise)
        net = layers.fully_connected(noise, 1024)
        logging.info(net)
        net = layers.fully_connected(net, 7 * 7 * 256)
        logging.info(net)
        net = tf.reshape(net, [-1, 7, 7, 256])
        logging.info(net)
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        logging.info(net)
        net = layers.conv2d_transpose(net, 32, [5, 5], stride=2, padding='VALID')
        logging.info(net)
        # TODO CHECK A PROPER WAY OF DOING THIS
        net = layers.conv2d_transpose(net, 32, [2, 2], stride=1, padding='VALID')
        logging.info(net)

        net = layers.conv2d(net, CHANNELS, 4, normalizer_fn=None, activation_fn=tf.tanh)
        logging.info(net)
    return net

def discriminator_fn(img, weight_decay=2.5e-5, is_training=True):
    """Discriminator network on MNIST digits.
    
    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        weight_decay: The L2 weight decay.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population 
            statistics.
    
    Returns:
        Logits for the probability that the image is real.
    """
    with framework.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        with framework.arg_scope([layers.batch_norm], is_training=is_training):
            net = layers.fully_connected(net, 1024, normalizer_fn=layers.batch_norm)
        return layers.linear(net, 1)

def model_fn(features, labels, mode, params):

    del features

    if mode == tf.estimator.ModeKeys.PREDICT:
        random_noise = features['noise']
        predictions = {
            'generated_images': generator_fn(noise, is_training=False)
        }

        return tpu.TPUEStimatorSpec(model=mode, predictions=predictions)

    batch_size = params['batch_size']
    images = features['images']
    noise = features['noise']

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    
    # Generate images
    g_images = generator_fn(noise=noise, is_training=is_training)

    # Pass on discriminator
    d_real = discriminator_fn(images, is_training=is_training)
    d_fake = discriminator_fn(g_images, is_training=is_training)

    # Compute discriminator loss
    loss_real = tf.nn.sigmoid_cross_entropy_with_logits(
        labels = tf.ones_like(d_real),
        logits = d_real
        )
    loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(
        labels = tf.zeros_like(d_fake),
        logits = d_fake
        )

    d_loss = loss_real + loss_fake

    # Compute generator loss
    g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels = tf.ones_like(d_fake),
        logits = d_fake
    )

    if mode == tf.estimator.ModeKeys.TRAIN:
        d_loss = tf.reduce_mean(d_loss)
        d_optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate, beta1=0.5)
        g_loss = tf.reduce_mean(g_loss)
        g_optimizer = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate, beta1=0.5)

        if FLAGS.use_tpu:
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
        def _eval_metric_fn(d_loss, g_loss):
            # When using TPUs, this function is run on a different machine than the
            # rest of the model_fn and should not capture any Tensors defined there
            return {
                'discriminator_loss': tf.metrics.mean(d_loss),
                'generator_loss': tf.metrics.mean(g_loss)}

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=tf.reduce_mean(g_loss),
            eval_metrics=(_eval_metric_fn, [d_loss, g_loss]))

    # Should never reach here
    raise ValueError('Invalid mode provided to model_fn')



logging.info('Start')
noise_dims = 64
# Create GAN estimator.
# gan_estimator = tfgan.estimator.GANEstimator(
#     model_dir='gs://iowa_bucket/cifar10/outputs/',
#     generator_fn=generator_fn,
#     discriminator_fn=discriminator_fn,
#     generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
#     discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
#     generator_optimizer=tf.train.AdamOptimizer(0.1, 0.5),
#     discriminator_optimizer=tf.train.AdamOptimizer(0.1, 0.5))

tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    FLAGS.tpu,
    zone=FLAGS.tpu_zone,
    project=FLAGS.gcp_project)

config = tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    model_dir=FLAGS.model_dir,
    tpu_config=tf.contrib.tpu.TPUConfig(
        num_shards=FLAGS.num_shards,
        iterations_per_loop=FLAGS.iterations_per_loop))

# TPU-based estimator used for TRAIN and EVAL
est = tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn,
    use_tpu=FLAGS.use_tpu,
    config=config,
    train_batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.batch_size)

# CPU-based estimator used for PREDICT (generating images)
cpu_est = tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn,
    use_tpu=False,
    config=config,
    predict_batch_size=_NUM_VIZ_IMAGES)

tf.gfile.MakeDirs(os.path.join(FLAGS.model_dir, 'generated_images'))


current_step = estimator._load_global_step_from_checkpoint_dir(FLAGS.model_dir)   # pylint: disable=protected-access,line-too-long
tf.logging.info('Starting training for %d steps, current step: %d' %
                (FLAGS.train_steps, current_step))
while current_step < FLAGS.train_steps:
    next_checkpoint = int(min(current_step + FLAGS.train_steps_per_eval,
                            FLAGS.train_steps))
    tf.logging.info('%s, %s,', type(next_checkpoint), next_checkpoint)
    est.train(input_fn=generate_input_fn(True),
                max_steps=next_checkpoint)
    current_step = next_checkpoint
    tf.logging.info('Finished training step %d' % current_step)

    if FLAGS.eval_loss:
        # Evaluate loss on test set
        metrics = est.evaluate(input_fn=generate_input_fn(False),
                                steps=dataset.NUM_EVAL_IMAGES // FLAGS.batch_size)
        tf.logging.info('Finished evaluating')
        tf.logging.info(metrics)

    # Render some generated images
    generated_iter = cpu_est.predict(input_fn=noise_input_fn)
    images = [p['generated_images'][:, :, :] for p in generated_iter]
    assert len(images) == _NUM_VIZ_IMAGES
    image_rows = [np.concatenate(images[i:i+10], axis=0)
                    for i in range(0, _NUM_VIZ_IMAGES, 10)]
    tiled_image = np.concatenate(image_rows, axis=1)

    img = dataset.convert_array_to_image(tiled_image)

    step_string = str(current_step).zfill(5)
    file_obj = tf.gfile.Open(
        os.path.join(FLAGS.model_dir,
                        'generated_images', 'gen_%s.png' % (step_string)), 'w')
    img.save(file_obj, format='png')
    tf.logging.info('Finished generating images')



# steps = 100000
# gan_estimator.train(input_fn, steps=steps)

# model = ExampleModel(tf_session=None,
#                         learning_rate=0.001,
#                         data_dir= 'gs://iowa_bucket/cifar10/data/',  # Dataset in GCloud Bucket
#                         use_tpu=True,
#                         output_path='gs://iowa_bucket/cifar10/outputs/'
# )

# logging.info('Build')
# gan_estimator.build_model()




# ### END
# logging.info('Train')
# gan_estimator.train(10001, input_fn)
