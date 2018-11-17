from absl import flags
from absl import logging
logging.set_verbosity(logging.INFO)

import tensorflow as tf

from core import DataManager
from model import ExampleModel
from datamanager import CIFAR10

# TODO This has to be externalized -> parse args
flags.DEFINE_bool('use_tpu', False, 'Use TPU model instead of CPU.')
flags.DEFINE_string('tpu', None, 'Name of the TPU to use.')
flags.DEFINE_string('data', None, 'Path to training and testing data.')  # Pass to data manager
flags.DEFINE_string(
    'model_dir', None,
    ('The directory where the model weights and training/evaluation summaries '
     'are stored.'))
flags.DEFINE_integer('batch_size', 32, 'Define the batch size input for the model.')


FLAGS = flags.FLAGS

batch_size = 32 # TESTING --> Not using the FLAGS yet
step_num = 100 # Number of steps (total number of samples used = batch_size * step)

with tf.Session() as session:
    dataset = CIFAR10(tf_session=session)

    model = ExampleModel(tf_session=session, data_manager=dataset, learning_rate=0.1)

    model.build_model(mode='train')

    model.train(501)
