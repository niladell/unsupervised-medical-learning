"""DISCLAIMER: OLD VERSION, NOT MAINTAINED AND PROBABLY NOT WORKING ANYMORE IN THIS VERSION.

For regular usage use 'main.py'. If for some reason you need to use a version not using tf.Estimator
get a 'model' and 'datamanager' from a version previous to the merge of 'devGan' This will be reincorporated 
in future releases."""

from absl import flags
from absl import logging
logging.set_verbosity(logging.INFO)

import tensorflow as tf

from core import DataManager
from model import ExampleModel
from datamanager import CIFAR10

# FLAGS --> NOT IN USE FOR NOW (needa copy them from main_tpu)

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
