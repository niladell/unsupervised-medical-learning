from absl import flags
from absl import logging
logging.set_verbosity(logging.INFO)

from os import walk
import pickle
import numpy as np

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver #pylint: disable=E0611

from core import DataManager
from model import ExampleModel
from datamanager import CIFAR10


model = ExampleModel(tf_session=None,
                        learning_rate=0.001,
                        data_dir= 'gs://iowa_bucket/cifar10/data/',  # Dataset in GCloud Bucket
                        use_tpu=True,
                        output_path='gs://iowa_bucket/cifar10/outputs/'
                    )

model.build_model()

### HIGHLY PROTOTIPY VERSON BELOW -- TODO MOVE TO DATA MANAGER

def _unpickle(file):
    """
    Unpicke batch CIFAR10 files

    Args:
        file (str): File name

    Returns:
        (dict): Extracted data from files
    """

    with open(file, 'rb') as fo:
        raw = pickle.load(fo, encoding='bytes')
    return raw

def _input_files(path, file_name):
    """
    Extract all the CIFAR-10 file names in the dataset folder.
    """
    logging.info(path)
    files = []
    for p,d,folder in walk(path):
        print(p,d,folder)
        for f in folder:
            if file_name in f:
                files.append(path + f)
    return files

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

def input_parser(file):
    """
    Parse CIFAR input files

    Returns:
        (np.array, list): Returns the images and the lables
    """

    # self._input_files()
    # images = None
    # labels = []
    # for i in self.files:
    raw = _unpickle(file) # -> dict_keys([b'batch_label', b'labels', b'data', b'filenames'])
    labels = raw[b'labels']
    images = _convert_images(raw[b'data'])
    return tf.cast(images, dtype=tf.float32), labels


def input_fn(params):
    """train_input_fn defines the input pipeline used for training."""
    batch_size = params["batch_size"]
    data_dir = params["data_dir"]
    # Retrieves the batch size for the current shard. The # of shards is
    # computed according to the input pipeline deployment. See
    # `tf.contrib.tpu.RunConfig` for details.
    image_files = _input_files(data_dir, 'data_batch')
    logging.info(image_files)
    
    # ds = Dataset.from_tensor_slices(input_parser(image_files[0]))
    # ds = Dataset.from_tensor_slices(image_files).interleave(
    #     lambda x: Dataset.from_tensor_slices(x).map(input_parser),
    #     cycle_length=1, block_length=1
    #     )
    ds = Dataset.from_tensor_slices((tf.random_normal([1000, 32, 32, 1], name='features'),
                                     tf.random_uniform([1000, 1], maxval=10, dtype=tf.int32, name='labels')))

    ds = ds.cache().repeat()
    ds = ds.batch(batch_size=batch_size, drop_remainder=True).prefetch(tf.contrib.data.AUTOTUNE)
    return ds #ds.make_initializable_iterator().get_next()

### END

model.train(10001, input_fn)