from absl import flags
from absl import logging
logging.set_verbosity(logging.INFO)

import tensorflow as tf

from core import DataManager
from model import ExampleModel
from core import DataManager

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
    # TODO Move all of this to the DataManager (UVP of using it vs vanilla tf.data.Dataset here in main)
    # Create a dataset tensor from the random data
    dataset = DataManager.from_tensor_slices((tf.random_uniform([1000, 28, 28, 1]), tf.cast(tf.round(tf.random_uniform([1000])), tf.int32)))
    # Automatically refill the data queue when empty
    dataset = dataset.repeat()
    # Create batches of data
    dataset = dataset.batch(batch_size)
    # Prefetch data for faster consumption
    dataset = dataset.prefetch(batch_size)
    iterator = dataset.make_initializable_iterator()
    # Initialize the iterator
    session.run(iterator.initializer)

    X, Y = iterator.get_next()

    model = ExampleModel(tf_session=session, data_manager=(X, Y), learning_rate=0.1)

    model.build_model(mode='train')

    model.train(100)

