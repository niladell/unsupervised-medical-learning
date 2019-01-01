"""Prototype run file

Example run command:
python src/run_tpu.py --model_dir=gs://iowa_bucket/cifar10/outputs__1 --data_dir=gs://iowa_bucket/cifar10/data  --tpu=node-1
"""

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
from PIL import Image


import tensorflow as tf
USE_ALTERNATIVE = False
try:
  from tensorflow.data import Dataset
except:
  Dataset = tf.data.Dataset
  USE_ALTERNATIVE = True
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
import cifar_model as model

SAMPLE_NUM = 50000

HEIGHT = WIDTH = 32
CHANNELS = 3

BATCH_SIZE = 128
NOISE_DIMS = 64
_NUM_VIZ_IMAGES = 100   # For generating a 10x10 grid of generator samples
NUM_EVAL_IMAGES = 100 # TODO No clue what is this

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
    'tpu_zone', default='us-central1-f',
    help='GCE zone where the Cloud TPU is located in. If not specified, we '
    'will attempt to automatically detect the GCE project from metadata.')

# Model specific paramenters
flags.DEFINE_string('model_dir', '', 'Output model directory')
flags.DEFINE_string('data_dir', '', 'Dataset directory')
flags.DEFINE_integer('noise_dim', 64,
                     'Number of dimensions for the noise vector')
flags.DEFINE_integer('batch_size', 1024,
                     'Batch size for both generator and discriminator')
flags.DEFINE_integer('num_shards', None, 'Number of TPU chips')
flags.DEFINE_integer('train_steps', 50000, 'Number of training steps')
flags.DEFINE_integer('train_steps_per_eval', 5000,
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

def convert_array_to_image(array):
  """Converts a numpy array to a PIL Image and undoes any rescaling."""
#   img = Image.fromarray(array, mode='RGB')

  img = Image.fromarray(np.uint8((array + 1.0) / 2.0 * 255), mode='RGB')
  return img


def input_fn(params):
    """Read CIFAR input data from a TFRecord dataset.

    Function taken from tensorflow/tpu cifar_keras repo"""
    del params
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
        image.set_shape([CHANNELS * HEIGHT * WIDTH])
        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
                tf.transpose(tf.reshape(image, [CHANNELS, HEIGHT, WIDTH]), [1, 2, 0]),
                tf.float32) * (2. / 255) - 1

        label = tf.cast(features['label'], tf.int32)

        return image, label
    # TODO Pass all of this via on params
    # image_files = ['gs://iowa_bucket/cifar10/data/train.tfrecords']
    image_files = [os.path.join(FLAGS.data_dir, 'train.tfrecords')]
    tf.logging.info(image_files)
    dataset = tf.data.TFRecordDataset([image_files])
    dataset = dataset.map(parser, num_parallel_calls=batch_size)
    dataset = dataset.prefetch(4 * batch_size).cache().repeat()
    if USE_ALTERNATIVE:
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    else:
        dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(2)
    # return dataset
    images, labels = dataset.make_initializable_iterator().get_next()
    images, labels = dataset.make_one_shot_iterator().get_next()
    # TODO why initializable vs one_shot

    random_noise = tf.random_normal([batch_size, NOISE_DIMS])

    features = {
        'real_images': images,
        'random_noise': random_noise}

    return features, labels


def generate_input_fn(is_training):
  """Creates input_fn depending on whether the code is training or not."""
  return input_fn

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)


def noise_input_fn(params):
  """Input function for generating samples for PREDICT mode.

  Generates a single Tensor of fixed random noise. Use tf.data.Dataset to
  signal to the estimator when to terminate the generator returned by
  predict().

  Args:
    params: param `dict` passed by TPUEstimator.

  Returns:
    1-element `dict` containing the randomly generated noise.
  """
  params['batch_size'] = BATCH_SIZE
  np.random.seed(0)
  noise_dataset = tf.data.Dataset.from_tensors(tf.constant(
      np.random.randn(params['batch_size'], FLAGS.noise_dim), dtype=tf.float32))
  noise = noise_dataset.make_one_shot_iterator().get_next()
  return {'random_noise': noise}, None


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
        'generated_images': model.generator(random_noise, is_training=False)
    }

    return tf.contrib.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions)

  # Use params['batch_size'] for the batch size inside model_fn
  batch_size = params['batch_size']   # pylint: disable=unused-variable
  real_images = features['real_images']
  random_noise = features['random_noise']

  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  generated_images = model.generator(random_noise,
                                     is_training=is_training)

  # Get logits from discriminator
  d_on_data_logits = tf.squeeze(model.discriminator(real_images))
  d_on_g_logits = tf.squeeze(model.discriminator(generated_images))

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
        learning_rate=FLAGS.learning_rate, beta1=0.5)
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
    ########
    # EVAL #
    ########
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
tpu_cluster_resolver = None
if FLAGS.use_tpu:
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


def save_samples_from_data():
    # Get some sample images
    # CPU-based estimator used for PREDICT (generating images)
    data_sampler = tf.contrib.tpu.TPUEstimator(
        model_fn=lambda features, labels, mode, params: tpu.TPUEstimatorSpec(mode=mode, predictions=features['real_images']),
        use_tpu=False,
        config=config,
        predict_batch_size=_NUM_VIZ_IMAGES)

    sample_images = data_sampler.predict(input_fn=input_fn)
    # sample_images = input_fn(None)
    tf.logging.info('That ran')
    images = []
    for i in range(_NUM_VIZ_IMAGES):
        images.append(next(sample_images))

    image_rows = [np.concatenate(images[i:i+10], axis=0)
                for i in range(0, _NUM_VIZ_IMAGES, 10)]
    tiled_image = np.concatenate(image_rows, axis=1)
    img = convert_array_to_image(tiled_image)

    step_string = str(current_step).zfill(5)
    file_obj = tf.gfile.Open(
        os.path.join(FLAGS.model_dir,
                        'generated_images', 'sampled_data.png'), 'w')
    img.save(file_obj, format='png')
save_samples_from_data()


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
                                steps=NUM_EVAL_IMAGES // FLAGS.batch_size)
        tf.logging.info('Finished evaluating')
        tf.logging.info(metrics)

    # Render some generated images
    generated_iter = cpu_est.predict(input_fn=noise_input_fn)
    images = [p['generated_images'][:, :, :] for p in generated_iter]
    if len(images) != _NUM_VIZ_IMAGES:
        tf.logging.info('Made %s images (when it should have been %s',
            len(images), _NUM_VIZ_IMAGES)
        images = images[:_NUM_VIZ_IMAGES]

    # assert len(images) == _NUM_VIZ_IMAGES
    image_rows = [np.concatenate(images[i:i+10], axis=0)
                    for i in range(0, _NUM_VIZ_IMAGES, 10)]
    tiled_image = np.concatenate(image_rows, axis=1)

    img = convert_array_to_image(tiled_image)

    step_string = str(current_step).zfill(5)
    file_obj = tf.gfile.Open(
        os.path.join(FLAGS.model_dir,
                        'generated_images', 'gen_%s.png' % (step_string)), 'w')
    img.save(file_obj, format='png')
    tf.logging.info('Finished generating images')
