from absl import flags
from absl import logging
logging.set_verbosity(logging.INFO)

import tensorflow as tf

from core import DataManager
from model import ExampleModel
from datamanager import CIFAR10

# TODO This has to be externalized -> parse args
# FLAGS --> NOT IN USE FOR NOW

batch_size = 32 # TESTING --> Not using the FLAGS yet
step_num = 100 # Number of steps (total number of samples used = batch_size * step)

with tf.Session() as session:
    training_dataset = CIFAR10(tf_session=session, file_name='data_batch_')  # Data batches from 1 to 5
    validation_dataset = CIFAR10(tf_session=session, file_name='test_batch')

    model = ExampleModel(tf_session=session,
                         learning_rate=0.001,
                         training_dataset=training_dataset,
                         validation_dataset=validation_dataset
                        )

    model.build_model()

    model.train(10001)
