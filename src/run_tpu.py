from absl import flags
from absl import logging
import coloredlogs
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

from core import DataManager
from model import ExampleModel
from datamanager import CIFAR10

logging.info('Start')
model = ExampleModel(tf_session=None,
                        learning_rate=0.001,
                        data_dir= 'gs://iowa_bucket/cifar10/data/',  # Dataset in GCloud Bucket
                        use_tpu=True,
                        output_path='gs://iowa_bucket/cifar10/outputs/'
                    )
logging.info('Build')
model.build_model()

SAMPLE_NUM = 50000
CHANNELS = 3
SIDE = 32
### HIGHLY PROTOTIPY VERSON BELOW -- TODO MOVE TO DATA MANAGER

def _convert_images(raw):
    """
    -- From TF Tutorials --
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """
    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw, dtype=float) / 255.0
    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, 3, 32, 32]) #self.num_channels, self.img_size, self.img_size])
    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])
    return images

def input_parser(tfrecord):
    """
    Parse CIFAR input files

    Returns:
        (np.array, list): Returns the images and the lables
    """

    features = {
        'image':   tf.FixedLenFeature([], tf.string),
        'label':   tf.FixedLenFeature([], tf.int64),
        # 'channels': tf.FixedLenFeature([], tf.int64),
        # 'height':   tf.FixedLenFeature([], tf.int64),
        # 'width':    tf.FixedLenFeature([], tf.int64),
        # 'samples':  tf.FixedLenFeature([], tf.int64)
    }

    sample = tf.parse_single_example(tfrecord, features)
    # images = tf.decode_raw(sample['images'], out_type=tf.uint8)
    # images.set_shape([SIDE * SIDE * CHANNELS])
    # labels = tf.decode_raw(sample['labels'], out_type=tf.int64)
    # #images = tf.reshape(images, shape=[])
    # # images = tf.reshape(images, [sample['samples'], sample['channels'],
    # #                             sample['height'], sample['width']])
    # # images.set_shape([sample['samples'].values, sample['channels'].values,
    # #                  sample['height'].values, sample['width'].values])
    # images = tf.reshape(images, [SIDE, SIDE, CHANNELS])

    # images = tf.cast(images, tf.float32)
    # labels = tf.cast(tf.reshape(labels, shape=[1]), tf.int32)
    # # labels.set_shape([])
    # logging.info('TFRecord: {}, {} '.format(images, labels))

    image = tf.decode_raw(sample['image'], tf.uint8)
    image.set_shape([CHANNELS * SIDE * SIDE])

    # Reshape from [depth * height * width] to [depth, height, width].
    image = tf.cast(
        tf.transpose(tf.reshape(image, [SIDE, SIDE, CHANNELS]), [0 ,1, 2]),
        tf.float32)

    label = tf.cast(features['label'], tf.int32)
    # label = tf.cast(sample['labels'], tf.int32)

    #assert False
    return image, label

def set_shapes(batch_size, images, labels):
    """Statically set the batch_size dimension."""
    transpose_input = False # TODO Delete this thing
    if transpose_input:
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([None, None, None, batch_size])))
        labels.set_shape(labels.get_shape().merge_with(
            tf.TensorShape([batch_size])))
    else:
        # images.set_shape(images.get_shape().merge_with(
        #     tf.TensorShape([batch_size, None, None, None])))
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([batch_size, SIDE, SIDE, CHANNELS])))
        labels.set_shape(labels.get_shape().merge_with(
            tf.TensorShape([batch_size, None])))
    return images, labels

# TODO REFACTOR ALL
# def input_fn(params):
#     """train_input_fn defines the input pipeline used for training."""
#     batch_size = params["batch_size"]
#     data_dir = params["data_dir"]
#     # Retrieves the batch size for the current shard. The # of shards is
#     # computed according to the input pipeline deployment. See
#     # `tf.contrib.tpu.RunConfig` for details.
#     # image_files = _input_files(data_dir, 'data_batch')
#     image_files = ['gs://iowa_bucket/cifar-10-data/train.tfrecord']
#     logging.info('Image files {}'.format(image_files))
#     logging.debug(' format {} and inner {}'.format(type(image_files), type(image_files[0])))
#     #ds = Dataset.from_tensor_slices(image_files)
#     ds = tf.data.TFRecordDataset(image_files)
#     ds = ds.map(input_parser)
#     # ds = ds.map(lambda filename: tf.py_func(input_parser, [filename], [tf.float32, tf.int32]))
#     # ds = Dataset.from_tensor_slices(image_files).interleave(
#     #     lambda x: Dataset.from_tensor_slices(x).map(input_parser),
#     #     cycle_length=1, block_length=1
#     #     )
#     #ds = Dataset.from_tensor_slices((tf.random_normal([1000, 32, 32, 1], name='features'),
#     #                                 tf.random_uniform([1000, 1], maxval=10, dtype=tf.int32, name='labels')))

#     # ds = ds.cache().repeat()
#     ds = ds.batch(batch_size=batch_size, drop_remainder=True)

#     # ds = ds.map(functools.partial(set_shapes, batch_size))

#     # ds = ds.prefetch(tf.contrib.data.AUTOTUNE)
#     return ds.make_one_shot_iterator().get_next()
#     # iterator = ds.make_initializable_iterator()
#     # tf.train.SessionRunHook.begin(iterator.initializer)
#     # return iterator.get_next()

def input_fn(params):
  """Read CIFAR input data from a TFRecord dataset.
  
  Function taken from tensorflow/tpu cifar_keras repo"""
  del params
  batch_size = 128
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
    return image, label

  # TEMPORAL
  image_files = ['gs://iowa_bucket/cifar-10-data/train.tfrecords']

  dataset = tf.data.TFRecordDataset([image_files])
  dataset = dataset.map(parser, num_parallel_calls=batch_size)
  dataset = dataset.prefetch(4 * batch_size).cache().repeat()
  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(1)
  return dataset


### END
logging.info('Train')
model.train(10001, input_fn)
